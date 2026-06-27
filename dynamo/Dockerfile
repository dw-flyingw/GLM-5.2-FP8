# Custom Dynamo SGLang runtime for GLM-5.2-FP8.
#
# Why: GLM-5.2-FP8 (GlmMoeDsaForCausalLM, DeepSeek-style sparse attention) requires
# SGLang >= 0.5.13.post1 (per the model card). No published Dynamo sglang-runtime
# image ships that yet:
#   1.2.1-cuda13       -> SGLang 0.5.11        (q_b_proj shape mismatch at load)
#   1.3.0-dev.1-cuda13 -> SGLang 0.5.12.post1  (still too old)
# So we take the newest Dynamo dev image (keeps the Dynamo runtime + NIXL + sgl_kernel
# wiring) and bump only SGLang to 0.5.13.post1.
#
# Build:  docker build -t glm52-dynamo-sglang:0.5.13post1 -f Dockerfile .
ARG BASE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13
FROM ${BASE}

ARG SGLANG_VERSION=0.5.13.post1

# This network is behind a corporate proxy; pip in the build container needs it.
# Pass via --build-arg (see serve.sh). Build with --network=host so the internal
# proxy host is reachable.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Proxy scoped to this RUN only (not ENV) so it isn't baked into the runtime image.
# --no-deps: keep the base image's torch / flashinfer / sgl-kernel / nixl pins intact
# (0.5.12->0.5.13 is a point release; runtime import is verified after build).
# SGLang 0.5.13.post1 requires sglang-kernel >= 0.4.3 (base image ships 0.4.2.post2).
# The 0.4.3 cu13 wheel is ABI-compatible with the base torch (verified: common_ops imports).
RUN https_proxy=${HTTPS_PROXY} http_proxy=${HTTP_PROXY} no_proxy=${NO_PROXY} \
    pip install --no-deps --break-system-packages \
        "sglang==${SGLANG_VERSION}" "sglang-kernel==0.4.3" \
    && python3 -c "import importlib.metadata as m; \
sg=m.version('sglang'); sk=m.version('sglang-kernel'); \
assert sg.startswith('0.5.13'), sg; assert sk >= '0.4.3', sk; \
print('sglang', sg, '| sglang-kernel', sk)"
# Note: importing sgl_kernel loads GPU-only CUDA ops (common_ops), so we check
# versions via package metadata here; the ops are verified at container run time.
