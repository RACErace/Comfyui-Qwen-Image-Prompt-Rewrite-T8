import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

import torch

import comfy.utils

from .pe_runtime import (DEFAULT_EDIT, DEFAULT_T2I, SERVER, file_signature, local_models,
                         pick_mmproj, prepare_images, quoted_literals, resolve_model, strip_quoted_literals)


# Prompt generation runs for minutes with no other sign of life, so report
# long steps to both the ComfyUI console and the node progress bar. llama.cpp
# streams one chunk per token but only reports a real total at the end, so the
# bar starts small and grows until the generation halts.
PROGRESS_START_TOTAL = 2048        # roughly a short rewrite
PROGRESS_GROW_STEP = 2048          # extend by this much when the model runs on
PROGRESS_MAX_TOTAL = 24000         # the edit branch's max_tokens
PROGRESS_LOG_INTERVAL = 5.0        # seconds between console lines


ASPECT_RATIOS = ["auto", "1:1", "1:2", "2:3", "3:4", "4:5", "16:9",
                 "9:16", "21:9", "9:21", "5:4", "4:3", "2:1"]


def _resolved_language(selection, rewritten_prompt, user_prompt=None):
    if selection == "中文":
        return "zh"
    if selection == "English":
        return "en"
    exact_literals = set(quoted_literals(user_prompt)) if user_prompt is not None else None
    prose = strip_quoted_literals(rewritten_prompt, exact_literals)
    return "zh" if re.search(r"[\u4e00-\u9fff]", prose) else "en"


def _format_prompt(prompt, transparent_rgba, language):
    pieces = []
    if transparent_rgba:
        pieces.append("这是一张带有透明度的RGBA图像。" if language == "zh"
                      else "This is an RGBA image with transparency. ")
    body = prompt.strip()
    if transparent_rgba and body and body[-1] not in ".!?。！？":
        body += "。" if language == "zh" else "."
    pieces.append(body)
    if transparent_rgba:
        pieces.append("该图像具有alpha通道，背景是透明的。" if language == "zh"
                      else " The image has an alpha channel, and the background is transparent.")
    return "".join(pieces)


def _choices(vision=False):
    names = list(local_models(vision))
    if vision:
        return ["Auto"] + names
    return names or ["(no local GGUF found)"]


def _extract_thinking(raw):
    """Pull the reasoning block out of the raw model answer, if it emitted one."""
    match = re.search(r"<think>([\s\S]*?)</think>", raw or "")
    return (match.group(1).strip() if match else "")


def _as_image_list(image):
    """Split one Comfy IMAGE batch into the single-image tensors prepare_images wants."""
    if image is None:
        return []
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("image must be a ComfyUI IMAGE (B,H,W,3|4)")
    if image.shape[0] < 1:
        raise ValueError("image batch is empty")
    if image.shape[0] > 10:
        raise ValueError("At most 10 reference images are supported")
    return [image[index:index + 1] for index in range(image.shape[0])]


class QwenPERewrite:
    @classmethod
    def INPUT_TYPES(cls):
        models = _choices()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamic_prompts": True}),
                "task": (["auto", "t2i", "edit"], {"default": "auto"}),
                "max_length": ("INT", {"default": 24000, "min": 1, "max": 32768,
                                       "tooltip": "生成长度上限。思考块也算在内，截断会自动关掉思考重试一次。"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.000001}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
                "thinking": ("BOOLEAN", {"default": True,
                                         "tooltip": "官方 PE 依赖思考块；关掉更快但改写质量可能下降。"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto",
                                                 "tooltip": "auto 使用模型建议；指定比例将覆盖模型比例并控制 Canvas。"}),
                "output_language": (["auto", "中文", "English"], {"default": "auto",
                                                                    "tooltip": "控制改写描述的语言；图内原文保留用户指定文字。"}),
                "transparent_rgba": ("BOOLEAN", {"default": False,
                                                    "tooltip": "将 RGBA、alpha 通道和透明背景要求加入最终提示词。"}),
                "t2i_model": (models, {"default": DEFAULT_T2I if DEFAULT_T2I in models else models[0]}),
                "edit_model": (models, {"default": DEFAULT_EDIT if DEFAULT_EDIT in models else models[0]}),
                "vision_model": (_choices(True), {"default": "Auto"}),
                "model_lifetime": (["after_run", "keep_loaded"], {"default": "after_run"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "参考图。batch 会按顺序拆成 <image1>、<image2>…"}),
                "system_prompt": ("STRING", {"multiline": True, "force_input": True,
                                             "tooltip": "替换内置的 prompts/system_prompt_*.txt。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "PE_RESULT", "STRING")
    RETURN_NAMES = ("generated_text", "thinking", "pe_result", "diagnostics")
    FUNCTION = "rewrite"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    @classmethod
    def IS_CHANGED(cls, task, t2i_model, edit_model, vision_model, **kwargs):
        fingerprint = hashlib.sha256()
        root = Path(__file__).resolve().parent
        fingerprint.update((root / "pe_nodes.py").read_bytes())
        fingerprint.update((root / "pe_runtime.py").read_bytes())
        for template in ("system_prompt_t2i.txt", "system_prompt_edit.txt"):
            fingerprint.update((root / "prompts" / template).read_bytes())
        # Comfy calls IS_CHANGED before resolving linked IMAGE outputs. Linked
        # values arrive as None here, so auto must include both possible models.
        names = [t2i_model, edit_model] if task == "auto" else [t2i_model if task == "t2i" else edit_model]
        for name in names:
            try:
                path = resolve_model(name)
            except FileNotFoundError:
                if task != "auto":
                    raise
                fingerprint.update(f"unresolved-model:{name}".encode())
                continue
            fingerprint.update(repr(file_signature(path)).encode())
        if task in ("auto", "edit"):
            try:
                path = pick_mmproj(edit_model, vision_model)
            except (FileNotFoundError, ValueError):
                if task == "edit":
                    raise
                fingerprint.update(f"unresolved-vision:{edit_model}:{vision_model}".encode())
            else:
                fingerprint.update(repr(file_signature(path)).encode())
        return fingerprint.hexdigest()

    def rewrite(self, prompt, task, max_length, temperature, top_k, top_p, min_p,
                repetition_penalty, presence_penalty, seed, thinking,
                aspect_ratio, output_language, transparent_rgba,
                t2i_model, edit_model, vision_model, model_lifetime,
                image=None, system_prompt=None):
        if not prompt.strip():
            raise ValueError("Enter a text instruction; image-only requests need an explicit editing goal")
        images = _as_image_list(image)
        actual_task = ("edit" if images else "t2i") if task == "auto" else task
        if actual_task == "edit" and not images:
            raise ValueError("edit requires at least one image")
        if actual_task == "t2i" and images:
            raise ValueError("t2i cannot receive images; use edit for image-conditioned generation")
        model_name = t2i_model if actual_task == "t2i" else edit_model
        model = resolve_model(model_name)
        mmproj = pick_mmproj(model_name, vision_model) if actual_task == "edit" else None
        encoded, dimensions = prepare_images(images)
        image_fingerprints = [hashlib.sha256(value.encode("ascii")).hexdigest() for value in encoded]
        context = 24576 if not images else (49152 if len(images) <= 5 else 65536)
        try:
            import comfy.model_management as memory
            memory.free_memory(12 * 1024**3, memory.get_torch_device())
            memory.soft_empty_cache()
        except ImportError:
            pass
        started = time.monotonic()
        logger = logging.getLogger("qwen_pe")
        progress = comfy.utils.ProgressBar(PROGRESS_START_TOTAL)
        log_state = {"tokens": 0, "last": 0.0, "gen_started": None}

        def on_token(_text, usage=None):
            reported = usage or {}
            total_tokens = reported.get("completion_tokens")
            if "chunks" in reported:
                # streaming: one chunk per token, no running total yet
                total = progress.total
                if total_tokens is None or total_tokens > total:
                    total = min(PROGRESS_MAX_TOTAL, max(PROGRESS_START_TOTAL,
                                                        total + PROGRESS_GROW_STEP))
                log_state["tokens"] = total_tokens if total_tokens is not None else log_state["tokens"] + 1
                progress.update_absolute(log_state["tokens"], total=total)
            else:
                log_state["tokens"] = total_tokens if total_tokens is not None else log_state["tokens"] + 1
                total = progress.total
                if log_state["tokens"] > total:
                    total = min(PROGRESS_MAX_TOTAL, max(PROGRESS_START_TOTAL, log_state["tokens"]))
                progress.update_absolute(min(log_state["tokens"], total), total=total)
            now = time.monotonic()
            if now - log_state["last"] >= PROGRESS_LOG_INTERVAL:
                log_state["last"] = now
                logger.info("[Qwen PE] 生成中: 约 %d tokens (%.0fs)",
                            log_state["tokens"], now - started)

        logger.info("[Qwen PE] 任务=%s 模型=%s 图片=%d 上下文=%d (%.1fs 预处理)",
                    actual_task, Path(model_name).name, len(images), context,
                    time.monotonic() - started)
        with SERVER.lock:
            try:
                logger.info("[Qwen PE] 启动 llama-server 并加载模型…")
                load_started = time.monotonic()
                SERVER.start(model, mmproj, context, 99)
                load_seconds = time.monotonic() - load_started
                log_state["gen_started"] = time.monotonic()
                logger.info("[Qwen PE] 模型就绪 (%.1fs)，开始生成提示词…", load_seconds)
                answer, info = SERVER.complete(actual_task, prompt, encoded, seed, 900,
                                               on_token=on_token,
                                               output_language=output_language,
                                               aspect_ratio=aspect_ratio,
                                               transparent_rgba=transparent_rgba,
                                               system_prompt=system_prompt or None,
                                               max_tokens=max_length,
                                               temperature=temperature, top_k=top_k, top_p=top_p,
                                               min_p=min_p, repetition_penalty=repetition_penalty,
                                               presence_penalty=presence_penalty,
                                               thinking=thinking)
            finally:
                if model_lifetime == "after_run":
                    SERVER.stop()
        usage = (info or {}).get("usage") or {}
        elapsed = time.monotonic() - started
        gen_seconds = elapsed - (log_state["gen_started"] - started if log_state["gen_started"] else 0)
        if usage.get("chunks"):
            logger.info("[Qwen PE] 生成结束: %d tokens, %.1f tok/s, finish_reason=%s, 总耗时 %.1fs",
                        usage["chunks"], usage["chunks"] / max(gen_seconds, 1e-6),
                        (info or {}).get("finish_reason"), elapsed)
        else:
            logger.info("[Qwen PE] 生成结束: finish_reason=%s, 总耗时 %.1fs",
                        (info or {}).get("finish_reason"), elapsed)
        if progress.total > 0:
            progress.update_absolute(progress.total, total=progress.total)
        model_wh_ratio = answer["wh_ratio"]
        model_ratio_follow = answer.get("ratio_follow", "")
        if aspect_ratio != "auto":
            answer["wh_ratio"] = aspect_ratio
            answer["ratio_follow"] = ""
        language = _resolved_language(output_language, answer["rewritten_prompt"], prompt)
        final_prompt = _format_prompt(answer["rewritten_prompt"], transparent_rgba, language)
        template_name = "system_prompt_t2i.txt" if actual_task == "t2i" else "system_prompt_edit.txt"
        if system_prompt and system_prompt.strip():
            system_prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
            system_prompt_source = "input"
        else:
            system_prompt_sha256 = hashlib.sha256(
                (Path(__file__).resolve().parent / "prompts" / template_name).read_bytes()).hexdigest()
            system_prompt_source = template_name
        reasoning = _extract_thinking(info.get("raw_output") or "")
        result = {
            "task": actual_task,
            "rewritten_prompt": final_prompt,
            "thinking": reasoning,
            "wh_ratio": answer["wh_ratio"],
            "ratio_follow": answer.get("ratio_follow", ""),
            "model_wh_ratio": model_wh_ratio,
            "model_ratio_follow": model_ratio_follow,
            "selected_aspect_ratio": aspect_ratio,
            "output_language": language,
            "transparent_rgba": transparent_rgba,
            "image_dimensions": dimensions,
            "image_count": len(images),
            "image_fingerprints": image_fingerprints,
            "model": model.name,
            "mmproj": mmproj.name if mmproj else "",
            "system_prompt_sha256": system_prompt_sha256,
            "system_prompt_source": system_prompt_source,
            "sampling": {"max_length": max_length, "temperature": temperature, "top_k": top_k,
                         "top_p": top_p, "min_p": min_p, "repetition_penalty": repetition_penalty,
                         "presence_penalty": presence_penalty, "thinking": thinking},
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "seed": seed,
            "finish_reason": info["finish_reason"],
            "usage": info["usage"],
            "format_retries": info.get("format_retries", 0),
            "first_format_error": info.get("first_format_error"),
            "truncation_retry": info.get("truncation_retry", False),
            "normalized_single_image_tags": info.get("normalized_single_image_tags", 0),
            "removed_background_sentences": info.get("removed_background_sentences", 0),
            "normalized_background_phrases": info.get("normalized_background_phrases", 0),
            "normalized_margin_phrases": info.get("normalized_margin_phrases", 0),
            "translation_fallback": info.get("translation_fallback", False),
            "translation_usage": info.get("translation_usage"),
        }
        diagnostics = json.dumps({key: value for key, value in result.items()
                                  if key not in ("rewritten_prompt", "thinking")},
                                 ensure_ascii=False)
        return result["rewritten_prompt"], reasoning, result, diagnostics


class QwenPECanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pe_result": ("PE_RESULT",),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32,
                                   "tooltip": "画布像素预算为 resolution²；跟随原图尺寸时，大图会等比缩至预算内。"}),
            "follow_input_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("INT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("width", "height", "latent", "ratio_source")
    FUNCTION = "canvas"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    def canvas(self, pe_result, resolution, follow_input_size):
        follow = pe_result["ratio_follow"]
        if follow:
            index = int(follow.removeprefix("<image").removesuffix(">")) - 1
            width, height = pe_result["image_dimensions"][index]
            source = follow
        else:
            left, right = map(int, pe_result["wh_ratio"].split(":"))
            ratio = left / right
            width = math.sqrt(resolution * resolution * ratio)
            height = math.sqrt(resolution * resolution / ratio)
            source = pe_result["wh_ratio"]
        if follow and not follow_input_size:
            ratio = width / height
            width = math.sqrt(resolution * resolution * ratio)
            height = math.sqrt(resolution * resolution / ratio)
        scale = min(1.0, resolution / math.sqrt(width * height), 4096 / max(width, height))
        width *= scale
        height *= scale
        if min(width, height) < 16:
            upscale = 16 / min(width, height)
            if max(width, height) * upscale > 4096 or width * height * upscale**2 > resolution**2:
                raise ValueError("aspect ratio is too extreme for the selected canvas budget")
            width *= upscale
            height *= upscale
        # Flooring to the required 16-pixel grid keeps scaled input canvases
        # inside the selected pixel budget instead of rounding back above it.
        width = max(16, math.floor(width / 16) * 16)
        height = max(16, math.floor(height / 16) * 16)
        import comfy.model_management as memory
        latent = torch.zeros([1, 64, height // 16, width // 16], device=memory.intermediate_device())
        return width, height, {"samples": latent}, source


class QwenPEUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pe_result": ("PE_RESULT",)}}

    RETURN_TYPES = ("PE_RESULT",)
    RETURN_NAMES = ("pe_result",)
    FUNCTION = "unload"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def unload(self, pe_result):
        SERVER.stop()
        return (pe_result,)


class QwenPEModelList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_models",)
    FUNCTION = "list_models"
    CATEGORY = "Qwen Image 2.1/Prompt Rewrite"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def list_models(self, refresh):
        models = local_models()
        vision = local_models(True)
        return (json.dumps({"models": list(models), "vision_models": list(vision)}, ensure_ascii=False, indent=2),)


NODE_CLASS_MAPPINGS = {
    "QwenPERewriteT8": QwenPERewrite,
    "QwenPECanvasT8": QwenPECanvas,
    "QwenPEUnloadT8": QwenPEUnload,
    "QwenPEModelListT8": QwenPEModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPERewriteT8": "Qwen Image 2.1 PE Rewrite T8",
    "QwenPECanvasT8": "Qwen Image 2.1 PE Canvas T8",
    "QwenPEUnloadT8": "Qwen Image 2.1 PE Unload T8",
    "QwenPEModelListT8": "Qwen Image 2.1 PE Local Models T8",
}
