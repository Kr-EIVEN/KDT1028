import React, { useState } from 'react';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [tags, setTags] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('http://127.0.0.1:5055/predict', {
      method: 'POST',
      body: formData,
    });


      const data = await res.json();

      if (!res.ok || data.error) {
        if (data.error?.includes('NSFW')) {
          setError("🚫 부적절한 사진입니다.");
        } else {
          setError("⚠️ 예측 중 오류가 발생했습니다.");
        }
        setTags(null);
        setPreviewUrl(null);
        return;
      }

      setTags(data);
      setError(null);
    } catch (err) {
      console.error("업로드 실패:", err);
      setError("❌ 서버 연결에 실패했습니다.");
      setTags(null);
      setPreviewUrl(null);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      handleUpload(file);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFileChange} />

      {error && (
        <div style={{ marginTop: '20px', color: 'red', fontWeight: 'bold' }}>
          {error}
        </div>
      )}

      {previewUrl && !error && (
        <div style={{ marginTop: '20px' }}>
          <img
            src={previewUrl}
            alt="Preview"
            style={{ maxWidth: '300px', borderRadius: '8px' }}
          />
        </div>
      )}

      {tags && tags.object_tags && (
  <div style={{ marginTop: '20px' }}>
    <h3>🎯 해시태그 결과</h3>
    <p><strong>Object:</strong> {tags.object_tags.join(' ')}</p>
    <p><strong>Scene:</strong> {tags.scene_tags.join(' ')}</p>
    <p><strong>Mood:</strong> {tags.mood_tags.join(' ')}</p>
  </div>
)}

{tags && tags.categories && (
  <div style={{ marginTop: '20px' }}>
    <h3>📊 카테고리 Top-3</h3>
    <ul>
      {tags.categories.map(([name, score], idx) => (
        <li key={idx}>
          {name}: {(score * 100).toFixed(1)}%
        </li>
      ))}
    </ul>
  </div>
)}

    </div>
  );
}

export default ImageUploader;