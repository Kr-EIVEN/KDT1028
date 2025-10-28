document.addEventListener('DOMContentLoaded', function() {
    var form = document.getElementById('login-form');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // 기본 제출 동작 방지

        var id = document.getElementsByName('id')[0].value;
        var pw = document.getElementsByName('passwd')[0].value;

        if (id === 'admin' && pw === 'admin1234') {
            alert("환영합니다.");
        } else {
            alert("다시 시도해주세요.");
        }
    });
});