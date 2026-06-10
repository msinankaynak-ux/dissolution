# DissolvA — Kontrol / Test / İyileştirme Prosedürü (Playbook)

> Ara ara çalıştırılan periyodik sağlık + iyileştirme turu.
> Her tur: kapsamı seç → adımları yürüt → bulguları kaydet → 1–2 iyileştirme uygula → push.
> Son güncelleme: 2026-06-10.

## Nasıl çalıştırılır
- Bana **"playbook'u çalıştır"** de. İstersen kapsam ver:
  - **Hızlı tur** → adım 1–3 (çalışırlık + bilimsel regresyon).
  - **Tam tur** → adım 1–7.
- Cowork her adımı klonda yürütür, sonunda **Bulgu + Aksiyon** listesi çıkarır,
  onayınla 1–2 iyileştirmeyi uygular, `dev`'e push eder.
- Her tur sonunda aşağıdaki **Run-Log**'a tarihli bir satır eklenir.
- Doğrulama ilkesi: hiçbir iyileştirme test edilmeden "tamam" sayılmaz
  (headless app testi / pytest / validasyon seti yeşil olmadan kapatılmaz).

## Adımlar (kaba hatlar)

### 1. Senkron & anlık görüntü
- 3 repoyu pull et; temiz mi, branch doğru mu (`tools/dissolva-status.sh`).
- O anki commit'leri/sürümü not al (geri dönüş noktası).

### 2. Çalışırlık / smoke test
- **Frontend:** headless `AppTest` — exception sayısı 0 mı.
- **Backend:** `/health` ayakta mı, `pytest` yeşil mi, ana uçlar
  (`/api/fit`, `/api/f2`, `/api/bootstrap-f2`, `/api/models`) yanıt veriyor mu.

### 3. Bilimsel doğruluk (regresyon)
- Validasyon setini yayınlanmış referanslara karşı koştur — 8/8 hâlâ geçiyor mu.
- **Engine paritesi:** `dissolva/models.py` ↔ backend `services/engine.py`
  aynı girdide aynı sonucu mu veriyor (byte-identik olmalı).
- Birkaç kinetik fit'i spot-check: K, n, R², f2 beklenen aralıkta mı.

### 4. Regülasyon uyumu
- f1/f2 + %85 tek-nokta kuralı, FDA CV erken-nokta kriteri, OOS/OOT flagleme doğru mu.
- Gerektiğinde kuralı birincil kılavuza (FDA 1997 / EMA CHMP-QWP / USP) karşı doğrula.

### 5. Güvenlik / IP
- Motor hâlâ **public** frontend'de mi (IP riski — kaldırma adımı açık).
- Secret/anahtar sızıntısı var mı; input bounds + XSS guard yerinde mi; bağımlılık zafiyeti.

### 6. UX / içerik
- "Tüm kullanıcı-metni İngilizce" kuralı bozulmuş mu.
- Grafik/etiket/eksen netliği, kırık link, literatür referansları güncel mi.

### 7. İyileştirme & kapanış
- Bulguları önceliklendir → **1–2 tanesini uygula** → doğrula (test) → `dev`'e push.
- Sağlamsa `dev` → `main` promote (production + Railway yeniden kurulur).
- Run-Log'a tarihli özet ekle.

## Run-Log
| Tarih | Kapsam | Bulgular | Uygulanan | Ertelenen |
|------------|--------|----------|-----------|-----------|
| _(ilk tur burada başlayacak)_ | | | | |
