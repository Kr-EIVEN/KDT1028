"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

export default function Navbar() {
  const [user, setUser] = useState<{ nickname: string } | null>(null);
  const pathname = usePathname();
  const router = useRouter();

  const isActive = (path: string) =>
    pathname === path
      ? "text-black font-semibold border-b-2 border-black pb-[1px]"
      : "hover:opacity-80 text-gray-700";

  // ✅ localStorage에서 로그인 정보 불러오기
  useEffect(() => {
    const stored = localStorage.getItem("user");
    if (stored) setUser(JSON.parse(stored));
  }, []);

  // ✅ 로그아웃 함수
  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
    router.push("/login");
  };

  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-white shadow-md">
      <div className="w-full px-8 lg:px-16">
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
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center">
            <Link href="/" className="inline-flex items-center gap-2">
              <Image
                src="/assets/imgs/cameraIcon.png"
                alt="PicShow"
                width={50}
                height={50}
                className="drop-shadow-md"
              />
              <span className="font-bold tracking-wide text-lg">PicShow</span>
            </Link>
          </div>

          {/* 오른쪽 메뉴 */}
          <div className="flex items-center gap-6 pr-6 lg:pr-10">
            <Link href="/explore" className={isActive("/explore")}>
              탐색
            </Link>

            <Link href="/customer" className={isActive("/customer")}>
              고객센터
            </Link>

            {/* ✅ 로그인 여부에 따른 표시 */}
            {user ? (
              <div className="flex items-center gap-3">
                <span className="font-semibold text-gray-800">
                  {user.nickname}님
                </span>
                <button
                  onClick={handleLogout}
                  className="text-sm px-3 py-1.5 rounded-md bg-gray-200 hover:bg-gray-300 text-gray-700 transition"
                >
                  로그아웃
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="px-3 py-1.5 rounded-md bg-gray-800 text-white text-sm font-medium hover:bg-gray-700 transition"
              >
                로그인 / 회원가입
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
