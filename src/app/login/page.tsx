"use client";

import { useState } from "react";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter(); // ✅ 반드시 함수 컴포넌트 안에 위치
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const [showSignup, setShowSignup] = useState(false);
  const [signupForm, setSignupForm] = useState({
    email: "",
    pw: "",
    pwCheck: "",
    nickname: "",
  });

  // ✅ 로그인 요청
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "로그인 실패");

      localStorage.setItem("user", JSON.stringify(data));
      router.push("/"); // ✅ 메인 페이지로 이동
    } catch (err: any) {
      setMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ✅ 회원가입 요청
  const handleSignupSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    if (signupForm.pw !== signupForm.pwCheck) {
      setMsg("비밀번호가 일치하지 않습니다.");
      return;
    }

    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: signupForm.email,
          password: signupForm.pw,
          nickname: signupForm.nickname,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "회원가입 실패");

      localStorage.setItem("user", JSON.stringify(data));
      router.push("/"); // ✅ 회원가입 후 메인으로
    } catch (err: any) {
      setMsg(err.message);
    }
  };

  return (
    <main className="bg-white min-h-screen text-gray-800 flex flex-col">
      <Navbar />

      <section className="flex-grow flex items-center justify-center py-24 bg-[#f9f9f9]">
        <div className="w-full max-w-lg bg-white shadow-md rounded-2xl p-8 border border-gray-100 relative">
          <h1 className="text-2xl font-bold text-center mb-6">로그인</h1>
          {msg && <p className="mb-4 text-sm text-center text-gray-700">{msg}</p>}
          <form onSubmit={handleSubmit} className="space-y-5">
            <input
              type="email"
              placeholder="이메일"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
              required
            />
            <input
              type="password"
              placeholder="비밀번호"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
              required
              minLength={6}
            />
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gray-900 text-white font-medium rounded-lg py-3 hover:bg-gray-800 transition disabled:opacity-60"
            >
              {loading ? "처리 중..." : "로그인"}
            </button>
          </form>

          <div className="text-center mt-6">
            <button
              onClick={() => setShowSignup(true)}
              className="text-gray-800 hover:text-gray-900 text-sm underline"
            >
              회원가입
            </button>
          </div>
        </div>
      </section>

      <Footer />

      {showSignup && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-lg w-full max-w-md p-8 relative">
            <h2 className="text-xl font-bold text-center mb-6">회원가입</h2>
            <button
              onClick={() => setShowSignup(false)}
              className="absolute top-4 right-5 text-gray-400 hover:text-gray-600 text-lg"
            >
              ✕
            </button>

            <form onSubmit={handleSignupSubmit} className="space-y-4">
              <input
                type="email"
                placeholder="이메일"
                value={signupForm.email}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, email: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
                required
              />
              <input
                type="text"
                placeholder="닉네임"
                value={signupForm.nickname}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, nickname: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
                required
              />
              <input
                type="password"
                placeholder="비밀번호"
                value={signupForm.pw}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, pw: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
                required
                minLength={6}
              />
              <input
                type="password"
                placeholder="비밀번호 확인"
                value={signupForm.pwCheck}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, pwCheck: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
                required
              />
              <button
                type="submit"
                className="w-full bg-gray-900 text-white font-medium rounded-lg py-3 hover:bg-gray-800 transition"
              >
                회원가입
              </button>
            </form>
          </div>
        </div>
      )}
    </main>
  );
}
