from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from nsfw_filter import detect_nsfw
from test import (
    load_resnet_ext_ckpt_or_fallback,
    predict_objects_over_threshold,
    load_clip_finetuned_or_fallback,
    clip_rank,
    SCENE_TAGS, MOOD_TAGS,
    prompt_scene, prompt_mood,
    OBJECT_THRESHOLD,
    run_infer
)

app = Flask(__name__)
CORS(app)

# 모델 로딩 (서버 시작 시 1회)
obj_model, obj_names, obj_tfm, _ = load_resnet_ext_ckpt_or_fallback("./runs_ft/object_ext")
clip_model, clip_pre, _ = load_clip_finetuned_or_fallback("./runs_ft/clip_finetune")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()  # ← 필터링용
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # NSFW 판별
    result = detect_nsfw(image_bytes)
    if result == "NSFW":
        return jsonify({'error': 'NSFW 이미지입니다. 해시태그 추출이 제한됩니다.'}), 403

    # 객체 해시태그
    obj_tags = predict_objects_over_threshold(image, obj_model, obj_names, obj_tfm, thr=OBJECT_THRESHOLD)
    # 장면/분위기 해시태그
    scene_tags = clip_rank(image, clip_model, clip_pre, SCENE_TAGS, prompt_scene)
    mood_tags  = clip_rank(image, clip_model, clip_pre, MOOD_TAGS,  prompt_mood)

    return jsonify({
        'object_tags': [t for t,_ in obj_tags],
        'scene_tags': [t for t,_ in scene_tags],
        'mood_tags':  [t for t,_ in mood_tags]
    })

if __name__ == '__main__':
    app.run(debug=True)
