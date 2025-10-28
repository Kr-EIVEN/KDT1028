"use client";
import { useState } from "react";
import Image from "next/image";

export default function Hero() {
  const [open, setOpen] = useState(false);
  return (
    <header className="relative h-[88vh] w-full">
      <Image
        src="/assets/imgs/header.jpg"
        alt="hero"
        fill
        priority
        className="object-cover"
      />
      <div className="absolute inset-0 bg-black/50" />

      <div className="relative z-10 max-w-4xl mx-auto px-4 h-full flex flex-col items-center justify-center text-center text-white">
        <h1 className="text-4xl md:text-6xl font-extrabold mb-4">
          픽쇼 방문이 처음이신가요?
        </h1>
        <p className="opacity-90 mb-8">
          일상에서 촬영한 사진들을 이웃과 공유하고 더 좋은사진을 위해 후원해주세요!!
        </p>

        <button
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-2 rounded-full border border-white/60 bg-white/10 px-6 py-3 hover:bg-white/20 transition"
        >
          {/* 아이콘 이미지 (public/assets/imgs/playicon.png) */}
          <img
            src="/assets/imgs/playicon.png"
            alt="Play Icon"
            className="w-5 h-5"
          />
          <span className="font-semibold text-white">Watch Video</span>
        </button>
      </div>

      {/* 모달 */}
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
