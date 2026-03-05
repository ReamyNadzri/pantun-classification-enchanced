"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export default function InsightsPage() {
    const [isDarkMode, setIsDarkMode] = useState(true);

    // Theme toggle
    useEffect(() => {
        document.documentElement.setAttribute(
            "data-theme",
            isDarkMode ? "dark" : "light"
        );
    }, [isDarkMode]);

    return (
        <>
            <header className="header">
                <div className="container header-content">
                    <Link href="/" style={{ textDecoration: 'none' }}>
                        <div className="logo cursor-pointer">
                            <span className="logo-icon">🪻</span>
                            <div>
                                <h1>Klasifikasi Pantun</h1>
                                <span className="logo-subtitle">Mengenal Pasti Tema Pantun</span>
                            </div>
                        </div>
                    </Link>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                        <Link href="/" className="nav-link">Klasifikasi</Link>
                        <Link href="/insights" className="nav-link active">Insights</Link>
                        <button
                            className="theme-toggle-btn"
                            onClick={() => setIsDarkMode(!isDarkMode)}
                            title={isDarkMode ? "Light Mode" : "Dark Mode"}
                        >
                            {isDarkMode ? "☀️" : "🌙"}
                        </button>
                    </div>
                </div>
            </header>

            <main>
                <div className="container" style={{ maxWidth: '800px', paddingBottom: '3rem' }}>
                    <section style={{ textAlign: 'center', marginBottom: '3rem' }}>
                        <h2 style={{
                            fontFamily: "'Outfit', sans-serif",
                            fontSize: '2.5rem',
                            background: 'linear-gradient(135deg, var(--primary), var(--accent))',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            marginBottom: '1rem'
                        }}>Model Insights</h2>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
                            Memahami bagaimana model-model Kecerdasan Buatan (AI) mengklasifikasikan pantun Melayu.
                        </p>
                    </section>

                    <div className="card" style={{ marginBottom: '2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '1rem' }}>
                            <h3 className="card-title" style={{ marginBottom: 0 }}>🏆 MalayBERT (Transformer)</h3>
                            <span className="theme-badge" style={{ padding: '0.4rem 1rem', fontSize: '0.9rem' }}>~60% F1-Score</span>
                        </div>

                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            <strong>Bagaimana ia berfungsi:</strong> Dibina di atas seni bina Transformer (<code style={{ color: 'var(--primary-light)' }}>mesolitica/bert-base-standard-bahasa-cased</code>), model ini tidak hanya mencari kata kunci tertentu. Ia membaca "keseluruhan konteks" pantun untuk memahami mesej atau niat penulis.
                        </p>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--primary)' }}>
                            <strong style={{ color: 'var(--primary-light)' }}>💡 Kekuatan & Kelemahan:</strong>
                            <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                Sangat hebat dalam memahami kiasan dan ibarat. Namun, kerana pantun Melayu sering bertujuan menyampaikan nasihat, MalayBERT cenderung untuk mengklasifikasikan hampir semua pantun moral sebagai <strong>Nasihat dan Pendidikan</strong>, walaupun ia menyentuh tentang Adat atau Alam.
                            </p>
                        </div>
                    </div>

                    <div className="card" style={{ marginBottom: '2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '1rem' }}>
                            <h3 className="card-title" style={{ marginBottom: 0 }}>🥈 SVM 90/10 (Machine Learning)</h3>
                            <span className="theme-badge" style={{ padding: '0.4rem 1rem', fontSize: '0.9rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border)', color: 'var(--text-primary)', boxShadow: 'none' }}>~55% F1-Score</span>
                        </div>

                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            <strong>Bagaimana ia berfungsi:</strong> Model klasik berasaskan algoritma Support Vector Machine (SVM) dan kaedah pengekstrakan ciri TF-IDF. Ia bergantung kepada pemadanan kekerapan perkataan (keyword frequency).
                        </p>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent)' }}>
                            <strong style={{ color: 'var(--accent-light)' }}>💡 Kekuatan & Kelemahan:</strong>
                            <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                Amat tepat jika terdapat kata kunci jelas seperti "Tuhan" (Agama) atau "Budi" (Budi). Akan tetapi, ia gagal menangkap nuansa. Jika pantun menyebut "Kasih sayang ibu", SVM terus menganggapnya pantun <strong>Percintaan</strong> semata-mata kerana wujudnya perkataan "Kasih".
                            </p>
                        </div>
                    </div>

                    <div className="card" style={{ marginBottom: '2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '1rem' }}>
                            <h3 className="card-title" style={{ marginBottom: 0 }}>🥉 TextCNN (Deep Learning)</h3>
                            <span className="theme-badge" style={{ padding: '0.4rem 1rem', fontSize: '0.9rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border)', color: 'var(--text-primary)', boxShadow: 'none' }}>~47% F1-Score</span>
                        </div>

                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            <strong>Bagaimana ia berfungsi:</strong> Convolutional Neural Network 1D (TextCNN) yang direka menggunakan PyTorch. Mengekstrak corak n-gram daripada vektor perkataan.
                        </p>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--text-muted)' }}>
                            <strong style={{ color: 'var(--text-primary)' }}>💡 Kekuatan & Kelemahan:</strong>
                            <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                Prestasinya paling lemah kerana rangkaian CNN memerlukan data berskala gergasi untuk membina filter yang bermakna. Oleh kerana dataset kita mempunyai kelas yang sangat kecil (seperti Teka-Teki dengan hanya 42 sampel ujian), TextCNN tidak mempunyai cukup data untuk "belajar".
                            </p>
                        </div>
                    </div>

                </div>
            </main>
        </>
    );
}
