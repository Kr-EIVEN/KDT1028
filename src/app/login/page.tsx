"use client";

import { useState } from "react";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showSignup, setShowSignup] = useState(false); // ✅ 회원가입 모달 상태
  const [signupForm, setSignupForm] = useState({
    id: "",
    pw: "",
    pwCheck: "",
    nickname: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert("로그인 기능은 아직 준비 중입니다.");
  };

  const handleSignupSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (signupForm.pw !== signupForm.pwCheck) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }
    alert("회원가입이 완료되었습니다!");
    setShowSignup(false);
  };

  return (
    <main className="bg-white min-h-screen text-gray-800 flex flex-col">
      <Navbar />

      {/* ✅ 로그인 폼 */}
      <section className="flex-grow flex items-center justify-center py-24 bg-[#f9f9f9]">
        <div className="w-full max-w-sm bg-white shadow-md rounded-2xl p-8 border border-gray-100 relative">
          <h1 className="text-2xl font-bold text-center mb-8">로그인</h1>

          <form onSubmit={handleSubmit} className="space-y-5">
            <input
              type="email"
              placeholder="이메일을 입력해 주세요"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
            />
            <input
              type="password"
              placeholder="비밀번호를 입력해 주세요"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
            />
            <button
              type="submit"
              className="w-full bg-gray-900 text-white font-medium rounded-lg py-3 hover:bg-gray-800 transition"
            >
              로그인
            </button>
          </form>

          {/* 하단 링크 */}
          <div className="flex justify-between items-center text-sm text-gray-500 mt-5">
            <label className="flex items-center gap-2">
              <input type="checkbox" className="accent-gray-700" />
              <span>아이디 저장</span>
            </label>
            <div className="space-x-2">
              <button className="hover:text-gray-800">아이디 찾기</button>
              <span>|</span>
              <button className="hover:text-gray-800">비밀번호 찾기</button>
            </div>
          </div>

          {/* ✅ 회원가입 버튼 */}
          <div className="text-center mt-6">
            <button
              onClick={() => setShowSignup(true)}
              className="text-gray-700 hover:text-gray-900 text-sm underline"
            >
              회원가입
            </button>
          </div>
        </div>
      </section>

      <Footer />

      {/* ✅ 회원가입 모달 */}
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
                type="text"
                placeholder="사용할 아이디를 입력해 주세요"
                value={signupForm.id}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, id: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
              />
              <input
                type="password"
                placeholder="비밀번호를 입력해 주세요"
                value={signupForm.pw}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, pw: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
              />
              <input
                type="password"
                placeholder="비밀번호를 다시 입력해 주세요"
                value={signupForm.pwCheck}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, pwCheck: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
              />
              <input
                type="text"
                placeholder="닉네임을 입력해 주세요"
                value={signupForm.nickname}
                onChange={(e) =>
                  setSignupForm({ ...signupForm, nickname: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
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

