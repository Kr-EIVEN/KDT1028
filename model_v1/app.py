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

# ✅ CORS 설정: 모든 origin 허용 + credentials 지원
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ 사람 태그 보강 함수 (단서 1개당 3% 확률 증가)
def enrich_person_tags(tags):
    object_tags = tags.get('object_tags', [])
    scene_tags = tags.get('scene_tags', [])
    mood_tags = tags.get('mood_tags', [])
    all_tags = object_tags + scene_tags + mood_tags

    person_clues = [
        "stadium", "soccer_field", "football_field", "court", "track","suit",
        "uniform", "jersey", "sportswear", "celebrating", "running", "jumping",
        "accessory", "bag", "watch", "glasses", "earrings", "necklace", "bracelet",
        "hoodie", "scarf", "sneakers", "shoes", "boots", "stage",
        "energetic", "bold", "dynamic","basketball","baseball","soccor","ballplayer"
    ]

    clue_count = sum(1 for tag in all_tags for clue in person_clues if clue in tag.lower())
    person_probability = clue_count * 0.03  # ✅ 단서 1개당 3% 확률 증가

    if person_probability >= 0.15:  # ✅ 15% 이상이면 태그 추가
        if not any("person" in tag.lower() for tag in all_tags):
            object_tags.append("#person")
        if not any("athlete" in tag.lower() for tag in all_tags):
            object_tags.append("#athlete")

    tags['object_tags'] = object_tags
    return tags

# ✅ 카테고리 보강 함수
def enrich_categories(tags):
    categories = tags.get('categories', [])
    
    flat_categories = [
        cat[0].lower() if isinstance(cat, tuple) else cat.lower()
        for cat in categories
    ]

    if any(tag in ['#person', '#athlete', '#human', '#people', '#man', '#woman'] for tag in tags.get('object_tags', [])):
        if not any("사람" in cat or "인물" in cat for cat in flat_categories):
            categories.append(("사람",))  

    tags['categories'] = categories
    return tags


# ✅ 예측 라우트
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ✅ NSFW 필터링
    if detect_nsfw(image_bytes) == "NSFW":
        return jsonify({'error': 'NSFW 이미지입니다. 해시태그 추출이 제한됩니다.'}), 403

    # ✅ 태그 추론 + 보강
    tags = run_full_infer(image)
    tags = enrich_person_tags(tags)
    tags = enrich_categories(tags)

    return jsonify({
        'object_tags': tags.get('object_tags', []),
        'scene_tags': tags.get('scene_tags', []),
        'mood_tags': tags.get('mood_tags', []),
        'categories': tags.get('categories', [])
    })

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5055)
