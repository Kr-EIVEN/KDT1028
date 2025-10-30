import React, { useState } from 'react';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [tags, setTags] = useState(null);
  const [selectedTags, setSelectedTags] = useState([]);
  const [confirmedTags, setConfirmedTags] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [confirmedCategories, setConfirmedCategories] = useState([]);

  function handleTagSelect(e) {
    const tag = e.target.value;
    if (e.target.checked) {
      setSelectedTags(prev => [...prev, tag]);
    } else {
      setSelectedTags(prev => prev.filter(t => t !== tag));
    }
  }

  function handleCategorySelect(e) {
    const cat = e.target.value;
    if (e.target.checked) {
      setSelectedCategories(prev => [...prev, cat]);
    } else {
      setSelectedCategories(prev => prev.filter(c => c !== cat));
    }
  }

  async function handleImageUpload(e) {
    const file = e.target.files[0];
    setImage(file);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('http://127.0.0.1:5055/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setTags(data);
      setSelectedTags([]);
      setConfirmedTags([]);
      setSelectedCategories([]);
      setConfirmedCategories([]);
    } catch (err) {
      console.error('서버 연결 실패:', err);
    }
  }

  return (
    <div style={{ padding: '20px' }}>
      <h2>이미지 업로드</h2>
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {image && (
        <div style={{ marginTop: '20px' }}>
          <img
            src={URL.createObjectURL(image)}
            alt="preview"
            style={{ maxWidth: '300px', borderRadius: '8px' }}
          />
        </div>
      )}

      {tags && (
        <>
          <div style={{ marginTop: '30px' }}>
            <h3>🎯 해시태그 선택</h3>
            {[...tags.object_tags, ...tags.scene_tags, ...tags.mood_tags].map((tag, idx) => (
              <label key={idx} style={{ marginRight: '10px' }}>
                <input
                  type="checkbox"
                  value={tag}
                  onChange={handleTagSelect}
                />
                {tag}
              </label>
            ))}

            <div style={{ marginTop: '20px' }}>
              <button onClick={() => setConfirmedTags(selectedTags)}>
                선택하기
              </button>
            </div>

            {confirmedTags.length > 0 && (
              <div style={{ marginTop: '20px' }}>
                <strong>선택된 해시태그:</strong> {confirmedTags.join(', ')}
              </div>
            )}
          </div>

          <div style={{ marginTop: '40px' }}>
            <h3>📂 카테고리 선택</h3>
            {tags.categories.map(([cat], idx) => (
              <label key={idx} style={{ marginRight: '10px' }}>
                <input
                  type="checkbox"
                  value={cat}
                  onChange={handleCategorySelect}
                />
                {cat}
              </label>
            ))}

            <div style={{ marginTop: '20px' }}>
              <button onClick={() => setConfirmedCategories(selectedCategories)}>
                선택하기
              </button>
            </div>

            {confirmedCategories.length > 0 && (
              <div style={{ marginTop: '20px' }}>
                <strong>선택된 카테고리:</strong> {confirmedCategories.join(', ')}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default ImageUploader;