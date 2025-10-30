from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# 🔧 모델 및 추론 관련 함수들
from nsfw_filter import detect_nsfw
from test import (
    load_clip,
    clip_encode_image,
    clip_encode_text,
    cosine_sim,
    softmax_normalize,
    infer_resnet_objects_raw,
    postprocess_objects,
    infer_objects,
    infer_scene,
    infer_mood_clip,
    infer_categories_clip,
    category_votes_from_tags,
    category_votes_from_scene_tags,
    mix_category_scores,
    run_full_infer
)

# ✅ Flask 앱 생성
app = Flask(__name__)

# ✅ CORS 설정: 모든 헤더, POST 메서드, localhost:5173 허용
CORS(app, origins=["http://localhost:5173"])

# ✅ 예측 라우트
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ✅ NSFW 필터링
    result = detect_nsfw(image_bytes)
    if result == "NSFW":
        return jsonify({'error': 'NSFW 이미지입니다. 해시태그 추출이 제한됩니다.'}), 403

    # ✅ 해시태그 추론
    tags = run_full_infer(image)

    return jsonify({
        'object_tags': tags.get('object_tags', []),
        'scene_tags': tags.get('scene_tags', []),
        'mood_tags': tags.get('mood_tags', []),
        'categories': tags.get('categories', [])
    })

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5055)
