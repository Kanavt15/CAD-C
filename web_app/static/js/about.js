// About Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Animate circular progress bars
    const circularProgressBars = document.querySelectorAll('.circular-progress');
    
    const progressObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                entry.target.classList.add('animated');
                const value = entry.target.dataset.value;
                const progressCircle = entry.target.querySelector('.progress');
                
                if (progressCircle) {
                    // Calculate stroke-dashoffset
                    const circumference = 2 * Math.PI * 90; // r=90
                    const offset = circumference - (circumference * value) / 100;
                    
                    // Add gradient definition if not exists
                    let svg = entry.target.querySelector('svg');
                    if (!svg.querySelector('defs')) {
                        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                        gradient.setAttribute('id', 'gradient');
                        gradient.setAttribute('x1', '0%');
                        gradient.setAttribute('y1', '0%');
                        gradient.setAttribute('x2', '100%');
                        gradient.setAttribute('y2', '100%');
                        
                        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                        stop1.setAttribute('offset', '0%');
                        stop1.setAttribute('stop-color', '#00d4ff');
                        
                        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                        stop2.setAttribute('offset', '100%');
                        stop2.setAttribute('stop-color', '#a855f7');
                        
                        gradient.appendChild(stop1);
                        gradient.appendChild(stop2);
                        defs.appendChild(gradient);
                        svg.prepend(defs);
                    }
                    
                    // Animate the progress
                    progressCircle.style.strokeDashoffset = circumference;
                    setTimeout(() => {
                        progressCircle.style.transition = 'stroke-dashoffset 2s ease-out';
                        progressCircle.style.strokeDashoffset = offset;
                    }, 100);
                }
            }
        });
    }, { threshold: 0.3 });
    
    circularProgressBars.forEach(bar => {
        progressObserver.observe(bar);
    });
    
    // Fade-in animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const fadeObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Apply fade-in to cards
    document.querySelectorAll('.tech-card, .performance-card, .use-case-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        fadeObserver.observe(el);
    });
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
});
