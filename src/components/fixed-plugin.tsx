"use client";
import Image from "next/image";
import { Button } from "@material-tailwind/react";

export function FixedPlugin() {
  return (
    <a
      href="https://www.material-tailwind.com"
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center rounded-md px-4 py-2 text-sm font-medium
                 bg-white text-gray-900 shadow hover:shadow-md transition"
    >
      Visit Material Tailwind
    </a>
  );
}

// ✅ default export는 반드시 컴포넌트 정의 **아래쪽**에 위치해야 합니다
export default FixedPlugin;
