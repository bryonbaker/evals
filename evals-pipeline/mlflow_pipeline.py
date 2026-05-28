"""
Kubeflow Pipeline for Automated Canopy Testing with MLflow

This pipeline implements an automated testing workflow that:
1. Clones a git repository
2. Scans directories for test configuration files
3. Calls the backend to generate responses
4. Evaluates responses using MLflow scorers
5. Uploads an HTML summary to S3
"""

import kfp
from typing import NamedTuple, List
from kfp import dsl
from kfp.dsl import component
from kfp import kubernetes


@component(base_image='python:3.9')
def git_clone_op(
    repo_url: str,
    branch: str = "main"
):
    """Clone a Git repository into the shared PVC."""
    import os
    import subprocess
    import shutil
    from urllib.parse import urlparse, urlunparse

    folder = "/prompts"

    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    username = os.getenv("GIT_USERNAME")
    password = os.getenv("GIT_PASSWORD")

    if username and password:
        parsed = urlparse(repo_url)
        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        repo_url = urlunparse(parsed._replace(netloc=netloc))

    print(f"Cloning {repo_url} at branch {branch} into {folder}")
    subprocess.run([
        "git", "clone",
        "--branch", branch,
        "--single-branch",
        "--depth", "1",
        repo_url,
        "/prompts"
    ], check=True)

    for item in os.listdir("/prompts"):
        print(item)


@component(base_image="python:3.9")
def scan_directory_op() -> NamedTuple("Output", [("configs", List[dict])]):
    """Scan /prompts for *_tests.yaml files."""
    import glob
    import os
    from collections import namedtuple

    configs = []
    base = "/prompts"

    for path in glob.glob(os.path.join(base, "**/**_tests.yaml"), recursive=True):
        rel_path = os.path.relpath(path, base)
        configs.append({"config_path": rel_path})

    print(f"Found {len(configs)} test config(s): {[c['config_path'] for c in configs]}")

    Output = namedtuple("Output", ["configs"])
    return Output(configs=configs)


@component(
    base_image="python:3.12",
    packages_to_install=["git+https://github.com/red-hat-data-services/mlflow@rhoai-3.4-ea.1", "httpx", "kubernetes", "litellm"]
)
def run_all_mlflow_tests(
    configs: List[dict],
    backend_url: str,
    llm_endpoint: str,
    mlflow_tracking_uri: str,
    git_hash: str = "test",
):
    """Call the backend, then evaluate responses with MLflow scorers."""
    import os

    # Set LLM env vars BEFORE importing mlflow/litellm so that litellm picks up
    # the correct base URL when it initialises its OpenAI client on first use.
    os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["OPENAI_API_KEY"] = "no-key-required"
    os.environ["OPENAI_API_BASE"] = llm_endpoint + "/v1"
    os.environ["OPENAI_BASE_URL"] = llm_endpoint + "/v1"
    os.environ["LLM_BASE_URL"] = llm_endpoint + "/v1"
    os.environ["LLM_API_KEY"] = "no-key-required"
    os.environ["LLM_MODEL"] = "llama32"

    import json
    import yaml
    import litellm
    import mlflow
    from typing import Literal
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import scorer, RetrievalGroundedness, RetrievalRelevance, ToolCallCorrectness, ToolCallEfficiency
    from urllib.parse import urljoin

    # Monkey-patch litellm module-level attributes so all LLM calls (including
    # built-in MLflow scorers that use the LiteLLM adapter) route to the local
    # vLLM endpoint rather than api.openai.com.
    litellm.api_base = llm_endpoint + "/v1"
    litellm.api_key = "no-key-required"

    # Monkey-patch _parse_chunk so that RETRIEVER spans returning plain strings
    # (instead of dicts with a "page_content" key) are handled correctly by the
    # RetrievalRelevance scorer.
    from mlflow.genai.utils import trace_utils as _trace_utils
    _orig_parse_chunk = _trace_utils._parse_chunk

    def _patched_parse_chunk(chunk):
        if isinstance(chunk, str):
            return {"content": chunk}
        return _orig_parse_chunk(chunk)

    _trace_utils._parse_chunk = _patched_parse_chunk

    namespace_path = "/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(namespace_path):
        with open(namespace_path) as f:
            os.environ["MLFLOW_WORKSPACE"] = f.read().strip()

    token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    @scorer
    def is_shorter(outputs: str, inputs: dict) -> bool:
        """Is the response shorter than the input?"""
        if "prompt" in inputs:
            return len(outputs) < len(inputs["prompt"])
        elif "messages" in inputs:
            user_content = " ".join(
                m.get("content", "") for m in inputs["messages"] if m.get("role") == "user"
            )
            return len(outputs) < len(user_content)
        return False

    # Backend helpers
    def send_request(payload, url):
        import httpx
        full_response = ""
        trace_id = None
        tool_calls = []
        with httpx.Client(timeout=None) as http_client:
            with http_client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = response.read().decode()
                    raise RuntimeError(f"Backend returned {response.status_code}: {error_body}")
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[len("data: "):])
                            full_response += data.get("delta", "")
                            if data.get("type") == "trace_id":
                                trace_id = data.get("trace_id")
                            elif data.get("type") == "tool_call":
                                tool_calls.append(data.get("name"))
                        except json.JSONDecodeError:
                            continue
        return full_response, trace_id, tool_calls

    def call_backend(inputs, endpoint):
        url = urljoin(backend_url, endpoint)
        if "prompt" in inputs:
            return send_request({"prompt": inputs["prompt"]}, url)
        elif "messages" in inputs:
            return send_request(inputs, url)
        return "", None, []

    def get_trace_with_retry(trace_id, retries=6, delay=3.0):
        """Fetch a trace, retrying to handle async trace ingestion lag."""
        import time
        for i in range(retries):
            trace = mlflow.get_trace(trace_id)
            if trace:
                return trace
            if i < retries - 1:
                print(f"  Trace {trace_id} not yet available, retrying in {delay}s ({i+1}/{retries-1})...")
                time.sleep(delay)
        return None

    def fetch_workspace_records(workspace, dataset_names):
        """Fetch records from named datasets in a given MLflow workspace."""
        original_workspace = os.environ.get("MLFLOW_WORKSPACE")
        os.environ["MLFLOW_WORKSPACE"] = workspace
        # Force the MLflow client to reinitialize so it picks up the new MLFLOW_WORKSPACE.
        # Calling set_tracking_uri with the same URI is often a no-op; using a dummy URI
        # first guarantees the tracking store is torn down and rebuilt with the new env var.
        mlflow.set_tracking_uri("")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        records = []
        try:
            for name in dataset_names:
                try:
                    dataset = mlflow.genai.get_dataset(name=name)
                    dataset_records = dataset.to_dict().get("records", [])
                    print(f"  Workspace '{workspace}', dataset '{name}': {len(dataset_records)} record(s)")
                    records.extend(dataset_records)
                except Exception as e:
                    print(f"  Workspace '{workspace}', dataset '{name}': not found ({e})")
        finally:
            if original_workspace is not None:
                os.environ["MLFLOW_WORKSPACE"] = original_workspace
            else:
                os.environ.pop("MLFLOW_WORKSPACE", None)
            mlflow.set_tracking_uri("")
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        return records

    # Derive sibling workspace names once, before the config loop
    current_workspace = os.environ.get("MLFLOW_WORKSPACE", "")
    base_name = current_workspace.rsplit("-", 1)[0]  # e.g. "user1-canopy" -> "user1"
    external_workspaces = [f"{base_name}-test", f"{base_name}-prod"]

    # Main loop
    repo_dir = "/prompts"

    for config_dict in configs:
        config_path = config_dict["config_path"]
        full_config_path = os.path.join(repo_dir, config_path)

        with open(full_config_path) as f:
            config = yaml.safe_load(f)

        usecase = config.get("usecase", config["name"])
        mlflow.set_experiment(usecase)

        endpoint = config["endpoint"]
        scorer_names = config.get("scorers", ["summary_quality", "is_shorter"])

        dataset_names = config.get("datasets", [])
        external_records = []
        for ws in external_workspaces:
            external_records.extend(fetch_workspace_records(ws, dataset_names))
        print(f"External records for '{config_path}': {len(external_records)}")

        # Load judge prompt from file if specified in config
        judge_prompt_file = config.get("judge_prompt")
        if judge_prompt_file:
            judge_prompt_path = os.path.join(os.path.dirname(full_config_path), judge_prompt_file)
            with open(judge_prompt_path) as f:
                judge_instructions = f.read()
        else:
            judge_instructions = (
                "{{ inputs }}\n{{ outputs }}\n{{ expectations }}\n"
                "Is the response accurate and consistent with the expected response? "
                "Respond with only \"yes\" or \"no\"."
            )

        summary_quality_judge = make_judge(
            name="summary_quality",
            instructions=judge_instructions,
            feedback_value_type=Literal["yes", "no"],
            model="openai:/llama32",
        )

        answer_quality_judge = make_judge(
            name="answer_quality",
            instructions=judge_instructions,
            feedback_value_type=Literal["yes", "no"],
            model="openai:/llama32",
        )

        retrieval_groundedness_scorer = RetrievalGroundedness(model="openai:/llama32")
        retrieval_relevance_scorer = RetrievalRelevance(model="openai:/llama32")
        tool_call_correctness_scorer = ToolCallCorrectness(model="openai:/llama32", should_exact_match=True)
        tool_call_efficiency_scorer = ToolCallEfficiency(model="openai:/llama32")

        SCORER_MAP = {
            "summary_quality": summary_quality_judge,
            "answer_quality": answer_quality_judge,
            "is_shorter": is_shorter,
            "tool_call_correctness": tool_call_correctness_scorer,
            "tool_call_efficiency": tool_call_efficiency_scorer,
        }

        RAG_SCORER_NAMES = {"retrieval_relevance", "retrieval_groundedness"}
        TOOL_TRACE_SCORER_NAMES = {"tool_call_correctness", "tool_call_efficiency"}
        text_scorers = [SCORER_MAP[n] for n in scorer_names if n in SCORER_MAP and n not in RAG_SCORER_NAMES and n not in TOOL_TRACE_SCORER_NAMES]
        rag_scorer_names = [n for n in scorer_names if n in RAG_SCORER_NAMES]
        tool_trace_scorer_names = [n for n in scorer_names if n in TOOL_TRACE_SCORER_NAMES]

        if not text_scorers and not rag_scorer_names and not tool_trace_scorer_names:
            print(f"Warning: no recognised scorers in {config_path}, skipping.")
            continue

        # Generate responses from backend
        eval_data = []
        for test in config.get("tests", []):
            inputs = test.get("inputs", {})
            expectations = test.get("expectations", {})
            if not inputs.get("prompt") and not inputs.get("messages"):
                continue
            print(f"Calling {endpoint} with test inputs...")
            generated, trace_id, tool_calls = call_backend(inputs, endpoint)
            entry_inputs = dict(inputs)
            if tool_calls:
                entry_inputs["tool_calls"] = tool_calls
            entry = {"inputs": entry_inputs, "outputs": generated, "expectations": expectations}
            if trace_id:
                entry["trace_id"] = trace_id
            eval_data.append(entry)

        # Call backend for external dataset records and append to eval_data
        for record in external_records:
            inputs = record.get("inputs", {})
            expectations = record.get("expectations", {})
            if not inputs.get("prompt") and not inputs.get("messages"):
                continue
            print(f"Calling {endpoint} with external record inputs...")
            generated, trace_id, tool_calls = call_backend(inputs, endpoint)
            entry_inputs = dict(inputs)
            if tool_calls:
                entry_inputs["tool_calls"] = tool_calls
            entry = {"inputs": entry_inputs, "outputs": generated, "expectations": expectations}
            if trace_id:
                entry["trace_id"] = trace_id
            eval_data.append(entry)

        if not eval_data:
            print(f"No test cases in {config_path}, skipping.")
            continue

        # Evaluate with MLflow
        print(f"Running MLflow evaluate for {config_path} with {len(eval_data)} test(s)...")
        with mlflow.start_run(run_name=f"{config['name']}_{git_hash}"):
            mlflow.log_param("config_path", config_path)
            mlflow.log_param("endpoint", endpoint)
            mlflow.log_param("git_hash", git_hash)

            all_scorers = list(text_scorers)

            # Fetch traces from external workspace once for both RAG and tool_call_correctness scorers.
            # Using a prompt->trace map avoids injecting internal keys into inputs (which would
            # pollute judge prompts) while still letting wrapper scorers look up the right trace.
            prompt_to_trace = {}
            if rag_scorer_names or tool_trace_scorer_names:
                trace_entries = [e for e in eval_data if "trace_id" in e]
                print(f"Fetching traces: {len(trace_entries)}/{len(eval_data)} entries have trace_id")
                if not trace_entries:
                    print("WARNING: no trace_ids in eval_data — backend may not be emitting trace_id SSE events.")
                else:
                    original_ws = os.environ.get("MLFLOW_WORKSPACE")
                    trace_id_to_trace = {}
                    for ws in external_workspaces:
                        os.environ["MLFLOW_WORKSPACE"] = ws
                        mlflow.set_tracking_uri("")
                        mlflow.set_tracking_uri(mlflow_tracking_uri)
                        try:
                            test_trace = get_trace_with_retry(trace_entries[0]["trace_id"])
                            if test_trace is None:
                                print(f"  Workspace '{ws}': first trace not found after retries, trying next.")
                                continue
                            print(f"  Workspace '{ws}': fetching {len(trace_entries)} trace(s)...")
                            for entry in trace_entries:
                                trace = get_trace_with_retry(entry["trace_id"])
                                if trace:
                                    trace_id_to_trace[entry["trace_id"]] = trace
                            print(f"  Fetched {len(trace_id_to_trace)} trace(s)")
                            break
                        except Exception as e:
                            print(f"  Workspace '{ws}': error — {e}")

                    if original_ws is not None:
                        os.environ["MLFLOW_WORKSPACE"] = original_ws
                    else:
                        os.environ.pop("MLFLOW_WORKSPACE", None)
                    mlflow.set_tracking_uri("")
                    mlflow.set_tracking_uri(mlflow_tracking_uri)

                    for entry in eval_data:
                        tid = entry.get("trace_id")
                        if tid and tid in trace_id_to_trace:
                            key = entry["inputs"].get("prompt") or str(entry["inputs"].get("messages", ""))
                            trace = trace_id_to_trace[tid]
                            prompt_to_trace[key] = trace
                            span_types = [getattr(s, "span_type", None) for s in (trace.data.spans or [])]
                            retriever_spans = [s for s in (trace.data.spans or []) if getattr(s, "span_type", None) == "RETRIEVER"]
                            print(f"  Trace {tid}: span_types={span_types}, retriever_spans={len(retriever_spans)}")
                            for rs in retriever_spans:
                                outputs = rs.outputs or []
                                n_chunks = len(outputs) if isinstance(outputs, list) else len(outputs.get("chunks", outputs))
                                print(f"    RETRIEVER span '{rs.name}': {n_chunks} chunk(s)")

            # RAG scorers — pass the pre-fetched trace directly to the built-in scorer
            if rag_scorer_names:
                if prompt_to_trace:
                    if "retrieval_groundedness" in rag_scorer_names:
                        @scorer
                        def retrieval_groundedness(inputs: dict):
                            key = inputs.get("prompt") or str(inputs.get("messages", ""))
                            trace = prompt_to_trace.get(key)
                            if not trace:
                                print(f"  retrieval_groundedness: no trace for key={key!r:.80}")
                                return None
                            result = retrieval_groundedness_scorer(trace=trace)
                            print(f"  retrieval_groundedness result: {result}")
                            return result
                        all_scorers.append(retrieval_groundedness)

                    if "retrieval_relevance" in rag_scorer_names:
                        @scorer
                        def retrieval_relevance(inputs: dict):
                            key = inputs.get("prompt") or str(inputs.get("messages", ""))
                            trace = prompt_to_trace.get(key)
                            if not trace:
                                print(f"  retrieval_relevance: no trace for key={key!r:.80}")
                                return None
                            result = retrieval_relevance_scorer(trace=trace)
                            print(f"  retrieval_relevance result: {result}")
                            return result
                        all_scorers.append(retrieval_relevance)
                else:
                    print(f"Warning: no traces found in {external_workspaces} for RAG scoring.")

            # Tool call correctness/efficiency wrappers — look up pre-fetched trace by prompt
            if tool_trace_scorer_names:
                if prompt_to_trace:
                    if "tool_call_correctness" in tool_trace_scorer_names:
                        @scorer
                        def tool_call_correctness(inputs: dict, expectations: dict):
                            key = inputs.get("prompt") or str(inputs.get("messages", ""))
                            trace = prompt_to_trace.get(key)
                            if not trace:
                                return None
                            expected = [{"name": n} for n in expectations.get("expected_tools", [])]
                            feedback = tool_call_correctness_scorer(
                                trace=trace,
                                expectations={"expected_tool_calls": expected},
                            )
                            return feedback.value if feedback else None
                        all_scorers.append(tool_call_correctness)

                    if "tool_call_efficiency" in tool_trace_scorer_names:
                        @scorer
                        def tool_call_efficiency(inputs: dict):
                            key = inputs.get("prompt") or str(inputs.get("messages", ""))
                            trace = prompt_to_trace.get(key)
                            if not trace:
                                return None
                            feedback = tool_call_efficiency_scorer(trace=trace)
                            return feedback.value if feedback else None
                        all_scorers.append(tool_call_efficiency)
                else:
                    print(f"Warning: no traces found in {external_workspaces} for tool trace scoring.")

            if all_scorers:
                results = mlflow.genai.evaluate(
                    data=eval_data,
                    scorers=all_scorers,
                )
                print(f"Eval metrics: {results.metrics}")

        print(f"Results logged to MLflow. Tracking URI: {mlflow_tracking_uri}")


@dsl.pipeline(
    name="Canopy Eval (MLflow)",
    description="Pipeline for running canopy evals with MLflow scoring"
)
def canopy_eval_pipeline(
    repo_url: str,
    branch: str = "main",
    backend_url: str = "",
    llm_endpoint: str = "",
    mlflow_tracking_uri: str = "",
    git_hash: str = "test",
):
    eval_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-eval-pvc",
        access_modes=["ReadWriteOnce"],
        size="3Gi",
        storage_class_name="gp3-csi",
    )

    # Step 1: Clone repo
    clone_task = git_clone_op(repo_url=repo_url, branch=branch)
    kubernetes.mount_pvc(clone_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")
    kubernetes.use_secret_as_env(
        clone_task,
        secret_name="git-auth",
        secret_key_to_env={"username": "GIT_USERNAME", "password": "GIT_PASSWORD"},
    )

    # Step 2: Scan for test configs
    scan_task = scan_directory_op()
    scan_task.after(clone_task)
    kubernetes.mount_pvc(scan_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")

    # Step 3: Run evaluations
    test_task = run_all_mlflow_tests(
        configs=scan_task.outputs["configs"],
        backend_url=backend_url,
        llm_endpoint=llm_endpoint,
        mlflow_tracking_uri=mlflow_tracking_uri,
        git_hash=git_hash,
    )
    test_task.after(scan_task)
    kubernetes.mount_pvc(test_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")


if __name__ == "__main__":
    arguments = {
        "repo_url":             "https://<USER_NAME>:<PASSWORD>@<GIT_SERVER>/<USER_NAME>/evals.git",  # 🚨 replace with your own repo URL
        "branch":               "main",
        "backend_url":          "http://canopy-backend:8000",
        "llm_endpoint":         "http://llama-32-predictor.ai501.svc.cluster.local:8080",
        "mlflow_tracking_uri":  "https://mlflow.redhat-ods-applications.svc.cluster.local:8443",
        "git_hash":             "test",
    }

    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path) as f:
        namespace = f.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"

    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path) as f:
        bearer_token = f.read()

    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )

    client.create_run_from_pipeline_func(
        canopy_eval_pipeline,
        arguments=arguments,
        experiment_name="kfp-evals-pipeline",
        enable_caching=False,
    )
