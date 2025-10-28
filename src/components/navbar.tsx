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
      ? "text-white font-semibold border-b-2 border-white pb-[1px]"
      : "hover:opacity-80";

  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-gradient-to-b from-black/60 to-transparent backdrop-blur-md transition">
      <div className="max-w-6xl mx-auto px-6">
        <ul className="flex items-center justify-between text-white text-[15px] h-[60px]">
          {/* 왼쪽 메뉴 */}
          <div className="flex items-center gap-8">
            <li><Link href="#" className="hover:opacity-90">이벤트</Link></li>
            <li><Link href="#" className="hover:opacity-90">공지사항</Link></li>
          </div>

          {/* 로고 중앙 */}
          <li className="flex items-center justify-center">
            <Link href="/" className="inline-flex items-center gap-2">
              <Image
                src="/assets/imgs/cameraIcon.png"
                alt="PicShow"
                width={28}
                height={28}
                className="drop-shadow-md"
              />
              <span className="font-bold tracking-wide text-lg hidden sm:inline">PicShow</span>
            </Link>
          </li>

          {/* 오른쪽 메뉴 */}
          <div className="flex items-center gap-6">
            <li><Link href="/explore" className={isActive("/explore")}>탐색</Link></li>

            {/* 검색 */}
            <li className="relative">
              <button
                className="inline-flex items-center gap-1 hover:opacity-90"
                onClick={() => setShowSearch((s) => !s)}
              >
                <span className="material-icons text-white/90">search</span>
                <span className="hidden sm:inline">검색</span>
              </button>

              {showSearch && (
                <div className="absolute right-0 mt-2 bg-white text-gray-800 rounded-lg shadow-lg p-3 w-64">
                  <input
                    className="w-full border rounded-md px-3 py-2 outline-none"
                    placeholder="검색어를 입력해 주세요."
                  />
                </div>
              )}
            </li>

            {/* 로그인 버튼 */}
            <li>
              <Link
                href="/login"
                className="px-3 py-1.5 rounded-md bg-white/20 hover:bg-white/30 text-sm font-medium"
              >
                로그인 / 회원가입
              </Link>
            </li>
          </div>
        </ul>
      </div>
    </nav>
  );
}
