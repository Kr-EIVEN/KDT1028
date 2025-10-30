"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import UploadModal from "@/components/UploadModal";

export default function Hero() {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false); // 임시 로그인 상태

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!search.trim()) {
      alert("검색어를 입력해주세요!");
      return;
    }
    alert(`검색어: ${search}`);
  };

  const handleUploadClick = () => {
    if (!isLoggedIn) {
      if (confirm("로그인이 필요합니다. 로그인 페이지로 이동하시겠습니까?")) {
        router.push("/login");
      }
    } else {
      setIsUploadOpen(true);
    }
  };

  return (
    <section
      className="relative h-[90vh] flex flex-col items-center justify-center text-center bg-cover bg-center"
      style={{ backgroundImage: "url('/assets/imgs/bg.jpg')" }}
    >
      {/* 어두운 오버레이 */}
      <div className="absolute inset-0 bg-black/30" />

      {/* 중앙 텍스트 */}
      <div className="relative z-10 text-white">
        <h1 className="text-4xl sm:text-5xl font-bold mb-4 drop-shadow-lg">
          픽쇼 방문이 처음이신가요?
        </h1>
        <p className="text-gray-100 mb-8 drop-shadow-md">
          일상에서 촬영한 사진들을 이웃과 공유하고 더 좋은 사진을 위해 후원해주세요!
        </p>

        {/* 🔍 검색창 */}
        <form
          onSubmit={handleSearch}
          className="flex items-center bg-white rounded-full shadow-md px-5 py-3 w-[400px] max-w-full mx-auto mb-5"
        >
          <input
            type="text"
            placeholder="사진을 검색해보세요..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-grow text-gray-700 outline-none"
          />
          <button type="submit" className="text-gray-500 hover:text-gray-700">
            🔍
          </button>
        </form>

        {/* 📸 사진 업로드 버튼 */}
        <button
          onClick={handleUploadClick}
          className="bg-white text-gray-900 px-6 py-3 rounded-full hover:bg-gray-100 transition shadow font-semibold"
        >
          사진 올리기
        </button>
      </div>

      {/* ✅ 업로드 팝업 */}
      <UploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} />
    </section>
  );
}

