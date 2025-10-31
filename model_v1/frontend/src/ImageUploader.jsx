import React, { useState } from 'react';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [tags, setTags] = useState(null);
  const [selectedTags, setSelectedTags] = useState([]);
  const [confirmedTags, setConfirmedTags] = useState([]);
  const [customTag, setCustomTag] = useState('');
  const [customTags, setCustomTags] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [confirmedCategories, setConfirmedCategories] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

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

  function handleCustomTagAdd() {
    const formatted = customTag.startsWith('#') ? customTag : `#${customTag}`;
    const allRecommended = [
      ...(tags?.object_tags || []),
      ...(tags?.scene_tags || []),
      ...(tags?.mood_tags || [])
    ];
    if (
      formatted &&
      !customTags.includes(formatted) &&
      !allRecommended.includes(formatted)
    ) {
      setCustomTags(prev => [...prev, formatted]);
      setSelectedTags(prev => [...prev, formatted]);
      setCustomTag('');
    }
  }

  async function handleImageUpload(e) {
    const file = e.target.files[0];
    setImage(file);
    setErrorMessage('');

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('http://127.0.0.1:5055/predict', {
        method: 'POST',
        body: formData,
      });

      if (res.status === 403) {
        setImage(null);
        setTags(null);
        setErrorMessage('🚫 부적절한 사진입니다.');
        return;
      }

      if (!res.ok) {
        throw new Error(`서버 응답 오류: ${res.status}`);
      }

      const data = await res.json();
      setTags(data);
      setSelectedTags([]);
      setConfirmedTags([]);
      setCustomTags([]);
      setCustomTag('');
      setSelectedCategories([]);
      setConfirmedCategories([]);
    } catch (err) {
      console.error('서버 연결 실패:', err);
      setErrorMessage('서버 연결에 실패했습니다.');
    }
  }

  return (
    <div style={{ padding: '20px' }}>
      <h2>이미지 업로드</h2>
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {errorMessage && (
        <div style={{
          marginTop: '20px',
          padding: '10px',
          backgroundColor: '#ffe6e6',
          color: '#cc0000',
          borderRadius: '8px',
          fontWeight: 'bold'
        }}>
          {errorMessage}
        </div>
      )}

      {image && (
        <div style={{ marginTop: '20px' }}>
          <img
            src={URL.createObjectURL(image)}
            alt="preview"
            style={{ maxWidth: '300px', borderRadius: '8px' }}
          />
        </div>
      )}

      {tags &&
        Array.isArray(tags.object_tags) &&
        Array.isArray(tags.scene_tags) &&
        Array.isArray(tags.mood_tags) &&
        Array.isArray(tags.categories) && (
          <>
            <div style={{ marginTop: '30px' }}>
              <h3>🎯 해시태그 선택</h3>
              {[...tags.object_tags, ...tags.scene_tags, ...tags.mood_tags].map((tag, idx) => (
                <label key={idx} style={{ marginRight: '10px' }}>
                  <input
                    type="checkbox"
                    value={tag}
                    checked={selectedTags.includes(tag)}
                    onChange={handleTagSelect}
                  />
                  {tag}
                </label>
              ))}

              {customTags.length > 0 && (
                <div style={{ marginTop: '10px' }}>
                  <h4>📝 직접 추가한 태그</h4>
                  {customTags.map((tag, idx) => (
                    <label key={`custom-${idx}`} style={{ marginRight: '10px' }}>
                      <input
                        type="checkbox"
                        value={tag}
                        checked={selectedTags.includes(tag)}
                        onChange={handleTagSelect}
                      />
                      {tag}
                    </label>
                  ))}
                </div>
              )}

              <div style={{ marginTop: '20px' }}>
                <input
                  type="text"
                  value={customTag}
                  onChange={e => setCustomTag(e.target.value)}
                  placeholder="직접 해시태그 입력"
                  style={{ marginRight: '10px' }}
                />
                <button onClick={handleCustomTagAdd}>추가하기</button>
              </div>

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
                    checked={selectedCategories.includes(cat)}
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