"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const [showSearch, setShowSearch] = useState(false);
  const pathname = usePathname();

  const isActive = (path: string) =>
    pathname === path
      ? "text-black font-semibold border-b-2 border-black pb-[1px]"
      : "hover:opacity-80 text-gray-700";

  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-[64px] text-gray-800">
          {/* 왼쪽 메뉴 */}
          <div className="flex items-center gap-8">
            <Link href="/event" className={isActive("/event")}>
              이벤트
            </Link>
            <Link href="/notice" className={isActive("/notice")}>
              공지사항
            </Link>
          </div>

          {/* 로고 중앙 */}
          <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center">
            <Link href="/" className="inline-flex items-center gap-2">
              <Image
                src="/assets/imgs/cameraIcon.png"
                alt="PicShow"
                width={28}
                height={28}
                className="drop-shadow-sm"
              />
              <span className="font-bold tracking-wide text-lg">PicShow</span>
            </Link>
          </div>

          {/* 오른쪽 메뉴 */}
          <div className="flex items-center gap-6">
            <Link href="/explore" className={isActive("/explore")}>
              탐색
            </Link>

            {/* 검색 */}
            <button
              className="flex items-center gap-1 hover:opacity-80"
              onClick={() => setShowSearch((s) => !s)}
            >
              <span className="material-icons text-gray-700">search</span>
              <span className="hidden sm:inline">검색</span>
            </button>

            {/* 로그인 버튼 */}
            <Link
              href="/login"
              className="px-3 py-1.5 rounded-md bg-gray-800 text-white text-sm font-medium hover:bg-gray-700 transition"
            >
              로그인 / 회원가입
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

