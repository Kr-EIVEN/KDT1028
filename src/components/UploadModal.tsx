"use client";

import { useState } from "react";

export default function UploadModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);

  if (!isOpen) return null;

  const handleUpload = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert("이미지를 선택해주세요!");
      return;
    }
    alert("이미지 업로드 완료! (테스트용)");
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 shadow-lg w-full max-w-md relative">
        <h2 className="text-xl font-bold text-center mb-6">사진 업로드</h2>
        <button
          onClick={onClose}
          className="absolute top-4 right-5 text-gray-400 hover:text-gray-600 text-lg"
        >
          ✕
        </button>

        <form onSubmit={handleUpload} className="flex flex-col items-center gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2"
          />
          <button
            type="submit"
            className="w-full bg-gray-900 text-white font-medium rounded-lg py-3 hover:bg-gray-800 transition"
          >
            업로드
          </button>
        </form>
      </div>
    </div>
  );
}
