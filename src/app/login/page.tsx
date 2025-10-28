"use client";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import Link from "next/link";

export default function LoginPage() {
  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert("데모: 로그인 처리 로직을 연결하세요.");
  };

  return (
    <main className="bg-gray-950 min-h-screen text-white">
      <Navbar />

      <section className="flex items-center justify-center py-20 px-4">
        <div className="w-full max-w-md rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-6">
          <h1 className="text-2xl font-bold mb-6 text-center">로그인</h1>
          <form onSubmit={onSubmit} className="grid gap-4">
            <label className="grid gap-2">
              <span className="text-sm opacity-90">아이디</span>
              <input className="px-3 py-2 rounded-md bg-white/10 border border-white/20 outline-none"
                     name="user_id" required />
            </label>
            <label className="grid gap-2">
              <span className="text-sm opacity-90">비밀번호</span>
              <input type="password" className="px-3 py-2 rounded-md bg-white/10 border border-white/20 outline-none"
                     name="password" required />
            </label>
            <button className="mt-2 rounded-md bg-white text-gray-900 font-semibold py-2 hover:opacity-90">
              로그인
            </button>
          </form>

          <div className="flex items-center justify-between text-sm mt-4">
            <a href="#" className="opacity-80 hover:opacity-100">비밀번호 찾기</a>
            <Link href="/signup" className="opacity-80 hover:opacity-100">회원가입</Link>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
