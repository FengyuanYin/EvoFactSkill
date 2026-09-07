(() => {
  const modules = [...document.querySelectorAll('.module')];
  const navItems = [...document.querySelectorAll('.nav-item')];
  const progress = document.querySelector('#progress-bar');

  navItems.forEach((button) => {
    button.addEventListener('click', () => {
      document.getElementById(button.dataset.target)?.scrollIntoView({ behavior: 'smooth' });
    });
  });

  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navItems.forEach((item) => item.setAttribute('aria-current', String(item.dataset.target === visible.target.id)));
  }, { rootMargin: '-30% 0px -60% 0px', threshold: [0, .25, .75] });
  modules.forEach((module) => observer.observe(module));

  const updateProgress = () => {
    const root = document.documentElement;
    const total = root.scrollHeight - root.clientHeight;
    const value = total > 0 ? Math.min(100, Math.max(0, root.scrollTop / total * 100)) : 0;
    if (progress) progress.style.width = `${value}%`;
  };
  addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  document.querySelectorAll('.quiz').forEach((quiz) => {
    const answer = quiz.dataset.answer;
    const feedback = quiz.querySelector('.quiz__feedback');
    quiz.querySelectorAll('button[data-choice]').forEach((button) => {
      button.addEventListener('click', () => {
        const correct = button.dataset.choice === answer;
        quiz.querySelectorAll('button[data-choice]').forEach((item) => item.removeAttribute('data-result'));
        button.dataset.result = correct ? 'correct' : 'wrong';
        if (feedback) feedback.textContent = correct ? (quiz.dataset.correct || 'Correct.') : (quiz.dataset.wrong || 'Not quite. Revisit the cited source and try again.');
      });
    });
  });
})();
