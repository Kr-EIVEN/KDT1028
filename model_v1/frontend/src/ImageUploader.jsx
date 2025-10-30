import React, { useState } from 'react';

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [tags, setTags] = useState(null);
  const [error, setError] = useState(null); // NSFW 메시지용

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        setError("🚫 부적절한 사진입니다."); // 경고 메시지 설정
        setTags(null);
        setPreviewUrl(null); // 이미지 미리보기 제거
        return;
      }

      setTags(data);
      setError(null); // 오류 초기화
    } catch (err) {
      console.error("업로드 실패:", err);
      setError("서버 연결에 실패했습니다.");
      setTags(null);
      setPreviewUrl(null);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null); // 이전 오류 초기화
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
    </div>
  );
}

export default ImageUploader;