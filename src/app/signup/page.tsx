"use client";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { useState } from "react";

export default function SignUpPage() {
  const [pw, setPw] = useState("");
  const [cpw, setCpw] = useState("");

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pw !== cpw) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }
    alert("데모: 회원가입 처리 로직을 연결하세요.");
  };

  return (
    <main className="bg-gray-950 min-h-screen text-white">
      <Navbar />

      <section className="flex items-center justify-center py-20 px-4">
        <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-6">
          <h1 className="text-2xl font-bold mb-6 text-center">회원가입</h1>

          <form onSubmit={onSubmit} className="grid gap-4">
            <div className="grid md:grid-cols-2 gap-4">
              <label className="grid gap-2">
                <span className="text-sm opacity-90">User ID</span>
                <input className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
              </label>
              <label className="grid gap-2">
                <span className="text-sm opacity-90">Name</span>
                <input className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
              </label>
            </div>

            <label className="grid gap-2">
              <span className="text-sm opacity-90">Username</span>
              <input className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
            </label>

            <label className="grid gap-2">
              <span className="text-sm opacity-90">Email</span>
              <input type="email" className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
            </label>

            <div className="grid md:grid-cols-2 gap-4">
              <label className="grid gap-2">
                <span className="text-sm opacity-90">Password</span>
                <input type="password" value={pw}
                       onChange={(e)=>setPw(e.target.value)}
                       className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
              </label>
              <label className="grid gap-2">
                <span className="text-sm opacity-90">Confirm Password</span>
                <input type="password" value={cpw}
                       onChange={(e)=>setCpw(e.target.value)}
                       className="px-3 py-2 rounded-md bg-white/10 border border-white/20" required />
              </label>
            </div>

            <button className="mt-2 rounded-md bg-white text-gray-900 font-semibold py-2 hover:opacity-90">
              회원가입
            </button>
          </form>
        </div>
      </section>

      <Footer />
    </main>
  );
}
