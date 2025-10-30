"use client";
import { useState } from "react";
import Image from "next/image";

export default function Hero() {
  const [open, setOpen] = useState(false);

  return (
    <header className="relative h-[88vh] w-full">
      {/* 배경 이미지 */}
      <Image
        src="/assets/imgs/main_page_img2.png"
        alt="hero"
        fill
        priority
        className="object-cover"
      />
      <div className="absolute inset-0 bg-black/40" />

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 h-full flex flex-col items-center justify-center text-center text-white">
        {/* 제목 */}
        <h1 className="text-3xl md:text-5xl font-extrabold mb-4 drop-shadow-lg">
          PicShow 방문이 처음이신가요?
        </h1>
        <p className="opacity-90 mb-10 text-sm md:text-lg">
          일상에서 촬영한 사진들을 이웃과 공유하고 더 좋은 사진을 위해 후원해주세요!
        </p>

        {/* 검색창 */}
        <div className="flex justify-center mb-8 w-full">
          <div className="bg-white rounded-full px-6 py-4 flex items-center shadow-lg w-[80%] md:w-[700px] lg:w-[800px]">
            <input
              type="text"
              placeholder="사진을 검색해보세요..."
              className="w-full text-gray-700 bg-transparent outline-none placeholder-gray-400 text-base md:text-lg"
            />
            <span className="material-icons text-gray-400 cursor-pointer text-xl md:text-2xl">
              search
            </span>
          </div>
        </div>

        {/* 버튼 그룹 */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-5">
          <button
            onClick={() => {
              const isLoggedIn = false; // 로그인 확인 로직
              if (!isLoggedIn) {
                if (confirm("로그인이 필요합니다. 로그인 페이지로 이동할까요?")) {
                  window.location.href = "/login";
                }
              } else {
                alert("사진 업로드 기능 실행!");
              }
            }}
            className="bg-white/90 text-gray-800 px-10 py-4 rounded-full font-semibold hover:bg-white transition shadow-lg text-base md:text-lg"
          >
            사진 올리기
          </button>

          <button
            onClick={() => (window.location.href = "/explore")}
            className="bg-white/30 text-white px-10 py-4 rounded-full font-semibold hover:bg-white/40 transition border border-white/40 text-base md:text-lg"
          >
            사진 탐색
          </button>
        </div>
      </div>

      {/* (선택) 비디오 모달 */}
      {open && (
        <div
          className="fixed inset-0 z-[60] bg-black/70 flex items-center justify-center p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="bg-black w-full max-w-3xl aspect-video rounded-xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <video controls autoPlay className="w-full h-full">
              <source src="/assets/imgs/tutorial-video.mp4" type="video/mp4" />
            </video>
          </div>
        </div>
      )}
    </header>
  );
}
