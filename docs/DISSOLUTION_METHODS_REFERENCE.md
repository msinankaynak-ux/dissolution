# DissolvA — Dissolüsyon Profili Değerlendirme Yöntemleri: Bilimsel Referans ve Doğrulama Planı

> Amaç: DissolvA'nın yaptığı hesapların **doğru, tutarlı, tekrarlanabilir ve eksiksiz** olduğunu
> güvence altına almak için, ilaç geliştirme ve kalite kontrolde kullanılan dissolüsyon profili
> değerlendirme ve karşılaştırma yöntemlerinin literatüre dayalı bir özeti ve buna bağlı bir
> doğrulama/aksiyon planı. Hazırlanma: 2026-06-10. Tüm bilimsel iddialar atıflandırılmıştır
> (scite/PubMed üzerinden çekilen makaleler; DOI'ler Referanslar bölümünde).

---

## 1. Bağlam: dissolüsyon testi neden kritik

Dissolüsyon testi hem yeni ürün geliştirmede hem de jeneriklerde temel bir kalite-kontrol
parametresidir; oral emilimi doğrudan etkilediği için biyoeşdeğerlik çalışmalarından önce
profillerin karşılaştırılması düzenleyici otoritelerce istenir (Endashaw et al., 2022; Zuo et al.,
2014). Profil karşılaştırma yaklaşımları üç ana gruba ayrılır: **(i) model-bağımsız**, **(ii)
model-bağımlı (kinetik)** ve **(iii) istatistiksel/çok-değişkenli** yöntemler (Costa & Sousa Lobo,
2001; Sahoo et al., 2008).

---

## 2. Model-bağımsız karşılaştırma: f1 ve f2

### 2.1 Tanım ve formüller
- **Fark faktörü (f1):** iki profil arasındaki yüzde farkının zaman noktaları üzerinden ortalaması.
- **Benzerlik faktörü (f2):**

  f2 = 50 · log{ [1 + (1/n) · Σ_{t=1}^{n} (R_t − T_t)²]^(−0.5) · 100 }

  burada n = zaman noktası sayısı, R_t / T_t = referans/test profilinin t anındaki ortalama %
  çözünmesi. f2 = 100 iki profil aynıyken; f2 azaldıkça farklılık artar (Islam, 2018).

### 2.2 Düzenleyici kabul ölçütü ve geçerlilik koşulları
- **f2 ≥ 50**, her zaman noktasında ortalama ≤ %10 farka karşılık gelir ve profiller "benzer"
  kabul edilir (Moellenhoff et al., 2018).
- FDA/f2 kuralları (Stevens et al., 2015):
  - Referans ve test için **12'şer birim**;
  - **en az 3 zaman noktası** (0 hariç);
  - hesaplamada **%85'in üzerinde yalnızca bir** çözünme değeri kullanılabilir;
  - Üç ortamda (0.1 N HCl, pH 4.5, pH 6.8) **15 dk içinde %85'ten fazla** çözünüyorsa f2'ye
    gerek yoktur ("çok hızlı çözünen" durum).
- Düşük varyans şartı: f2'nin ayırt edici gücü birimler-arası varyans yükseldikçe düşer; yöntem
  geliştirilirken varyans izlenmeli ve minimize edilmelidir (Stevens et al., 2015). Yaygın CV/RSD
  eşikleri: erken nokta (≤15 dk) için RSD ~%20, sonraki noktalar için ~%10 üstünde f2 doğrudan
  kullanılmamalıdır.

### 2.3 f2'nin bilinen sınırları
- f2 **noktasal (point-estimate)** bir ölçüdür; **örnekleme varyansını ve zaman noktaları arası
  korelasyonu yok sayar** (Zhai et al., 2016).
- f2 **profil şekline duyarsızdır** ve f2'nin istatistiksel dağılımı için kapalı bir formül
  olmaması en büyük zaafıdır (Zuo et al., 2014).
- Yüksek varyanslı verilerde f2 **yanlış-pozitif** ("benzer") sonuç üretebilir; böyle durumlarda
  alternatif model-bağımsız yöntemler gerekir (Romodanovsky & Goryachev, 2021).
- f2 yalnızca **ortalama** profilleri karşılaştırır; bireysel profiller için uygun değildir (Zhai
  et al., 2016).

---

## 3. Yüksek varyans durumu: Bootstrap f2

f2 sapmalı (biased) ve muhafazakâr bir tahmindir; bootstrap, f2 için bir **güven aralığı (GA)**
kurarak örnekleme varyansını hesaba katar (Shah et al., 1998).

- **Prosedür (BCa — bias-corrected and accelerated):** test ve referans için 12 birimden
  yer-değiştirmeli yeniden örnekleme ile N (örn. 1000) bootstrap örneği üret → her örnekte f2
  hesapla → bias-düzeltme istatistiği ve **jackknife (örn. n = 24)** ile ivme (acceleration)
  istatistiğini hesapla → GA'nın alt/üst sınırlarını çıkar; karar **alt GA sınırı**na göre verilir
  (Stevens et al., 2015; Islam, 2018).
- **Parametrik vs nonparametrik:** parametrik bootstrap çok-değişkenli normallik varsayar
  (ki-kare/normallik testiyle kontrol); nonparametrik bootstrap dağılımdan bağımsızdır (Islam,
  2018). Çarpık dağılımlarda BC ve bootstrap-t düzeltmeleri standart yüzdelik yöntemini iyileştirir.
- **Önemli ilke:** bootstrap, f2'yi "iyileştirmek" için değil, **varyans sorununu yönetmek** için
  kullanılır (Stevens et al., 2015). Bootstrap özellikle **f2 < 60** olduğunda kritik önemdedir
  (Zuo et al., 2014).
- **Tekrarlanabilirlik uyarısı:** bootstrap f2 sonucu kullanılan **yazılıma, varyansa, örneklem
  büyüklüğüne ve bootstrap sayısına** göre değişebilir (Boddu et al., 2023) — bu, bizim için
  doğrudan bir tutarlılık/tekrarlanabilirlik gereksinimidir (bkz. §6).

---

## 4. Diğer model-bağımsız ölçütler

- **Dissolüsyon Etkinliği (DE):** çözünme eğrisinin belirli bir (t1–t2) aralığındaki eğri-altı
  alanının, aynı sürede %100 çözünmeye karşılık gelen dikdörtgen alana oranı (Sahoo et al., 2008;
  Endashaw et al., 2022).
- **Ortalama Çözünme Süresi (MDT):** MDT = Σ(t_i · ΔQ_i) / Q_∞; salım hızını ve polimerin
  geciktirme etkinliğini özetler (Endashaw et al., 2022).
- İki ürün, DE değerleri ± %10 içinde ise eşdeğer kabul edilebilir (Endashaw et al., 2022).

## 4b. İleri istatistiksel/çok-değişkenli alternatifler
- **Çok-değişkenli güven bölgesi (Mahalanobis mesafesi):** birim-içi **CV > %15** olduğunda f2
  yerine önerilir (Zuo et al., 2014).
- **Maksimum sapma (maximum deviation) testi:** f2'nin geçerlilik koşulları sağlanmadığında bile
  uygulanabilen, ölçümlerin zamana göre varyansını da içeren bir EMA-uyumlu alternatif
  (Moellenhoff et al., 2018).
- **Tolerans limitleri:** f1/f2'yi örnekleme varyansı ve zaman-korelasyonunu içerecek şekilde
  sağlam istatistiksel temele oturtan yaklaşım (Zhai et al., 2016).
- Genel statistiksel karşılaştırma çerçeveleri için ayrıca Wang et al. (2015).

---

## 5. Model-bağımlı (kinetik) yöntemler

### 5.1 Model kütüphanesi
Higuchi denkleminden (1961) bu yana çok sayıda matematiksel model türetilmiştir; başlıcaları:
sıfır-derece, birinci-derece, Higuchi, Korsmeyer-Peppas (KP), Hixson-Crowell, Hopfenberg,
Baker-Lonsdale, Weibull, ve sigmoid/ampirik türevler (Costa & Sousa Lobo, 2001; Zhang et al., 2010).
Tüm modellerde F = t anında çözünen yüzdedir. Örnek parametreler (Zhang et al., 2010):
- **KP:** F = k_KP · t^n; k_KP yapısal/geometrik sabit, **n difüzyonel üs** (salım mekanizması
  göstergesi).
- **Hixson-Crowell:** çözünme sırasında yüzey alanı azalan partiküllerin salımı.
- **Weibull:** α = ölçek (zaman ölçeği), β = şekil parametresi (β=1 üstel; β>1 sigmoid;
  β<1 parabolik), T_i = lag/konum parametresi.
- **Baker-Lonsdale:** Higuchi'den türetilen, küresel matristen Fickian difüzyon.

### 5.2 KP "n" üssünün yorumu (mekanizma)
n değeri salım mekanizmasını ayırt eder ancak **dozaj formunun geometrisine** bağlıdır (Zuo et
al., 2014). Silindirik form (tablet) için: n ≤ 0.45 Fickian difüzyon; 0.45 < n < 0.89 anormal
(anomalous) taşıma — Fickian difüzyon + polimer relaksasyonu birleşimi; n ≈ 0.89 Case-II;
n > 0.89 süper Case-II (Siswanto et al., 2015). **Önemli:** KP modeli tipik olarak yalnızca ilk
**~%60 salım** için geçerlidir.

### 5.3 Model seçim kriterleri
- **R²_adjusted:** farklı parametre sayısına sahip modelleri karşılaştırmak için en uygun ölçüt;
  çünkü ham R² parametre eklendikçe daima artar, R²_adj ise aşırı-uyumda (over-fitting) düşebilir —
  en iyi model en yüksek R²_adj olandır (Zhang et al., 2010).
- **AIC (Akaike):** en düşük AIC en iyi modeli gösterir; veri büyüklüğüne ve nokta sayısına
  bağımlıdır (Zhang et al., 2010; Siswanto et al., 2015).
- **MSC (Model Selection Criterion):** MSC > 2–3 iyi uyum kabul edilir; veri büyüklüğünden bağımsız
  olduğu için modeller arası karşılaştırmada elverişlidir (Zhang et al., 2010).
- **Mekanistik makullük:** mekanistik modellerde seçim yalnızca uyum iyiliğine değil, modelin
  **mekanik olarak makul** olmasına da dayanmalıdır — kötü bir model bile iyi uyum verebilir
  (Zhang et al., 2010; Zuo et al., 2014).

### 5.4 Uyum (fitting) metodolojisi
- DDSolver, **doğrusal olmayan en küçük kareler** ve Nelder-Mead simpleks algoritmasıyla
  **dönüştürülmemiş** veriye uyum yapar (Zhang et al., 2010).
- **Kritik tutarlılık noktası:** KP gibi modellerde n değeri, **doğrusal-dönüşümlü** regresyon
  (log-log) ile **doğrusal-olmayan** regresyon arasında **farklı** çıkar; dönüşüm deneysel hatayı
  bozduğu için doğrusal-olmayan regresyon tercih edilir (Zuo et al., 2014). Bu, yazılımlar
  arasında gözlenen n/β farklılıklarının başlıca nedenidir ve bizim doğrulama setimizdeki Weibull
  β farkını da açıklayabilir.

---

## 6. DissolvA için doğrulama ve aksiyon planı

Aşağıdaki maddeler "doğru / tutarlı / tekrarlanabilir / eksiksiz" hedefini somut testlere çevirir.

**A. Doğruluk (literatüre/altın-standarda karşı)**
1. Yayınlanmış referans değerlerine (DDSolver: Zhang 2010; Zuo 2014 — f2 = 23.21/46.66/17.91)
   karşı f2 ve kinetik parametre paritesi koru ve genişlet.
2. Her modelin formülünü ve parametre tanımını Zhang et al. (2010) Tablo I ile bire bir doğrula.
3. f2 geçerlilik mantığını Stevens et al. (2015) kurallarına göre test et: 12 birim, ≥3 nokta,
   yalnızca tek >%85 noktası, 15 dk/%85 muafiyeti.

**B. Tutarlılık (motor ↔ backend ve dönüşüm tuzakları)**
4. `dissolva/models.py` ↔ backend `services/engine.py` byte-özdeş çıktı testi (aynı girdi → aynı
   sonuç) regresyon testi olarak sabitle.
5. KP "n" ve Weibull "β" için **yalnızca doğrusal-olmayan regresyon** kullanıldığını doğrula;
   log-dönüşümlü hesap varsa kaldır (Zuo et al., 2014).

**C. Tekrarlanabilirlik (özellikle bootstrap)**
6. Bootstrap f2 için **RNG seed sabitle**; N (örn. 1000), jackknife n, BCa düzeltmesini açıkça
   raporla — çünkü sonuç yazılım/N/örneklem'e duyarlıdır (Boddu et al., 2023).
7. Aynı veri + aynı ayarlarla tekrar koşumda **bit-düzeyinde aynı** f2 GA'sını üret (determinizm testi).

**D. Eksiksizlik (regülasyon ve uç durumlar)**
8. Yüksek varyans senaryosu: CV > %15'te f2 yerine bootstrap/çok-değişkenli yönlendirmesi yap
   (Zuo et al., 2014); CV eşiklerinde otomatik uyarı.
9. f2 sınır bölgesi (45–55) ve f2 < 60'ta bootstrap'ı öner (Zuo et al., 2014).
10. Mekanistik makullük uyarısı: en düşük AIC modeli mekanik olarak makul değilse kullanıcıyı
    uyar (Zhang et al., 2010).
11. KP geçerliliğini ~%60 salımla sınırla (otomatik not).

**E. Veri gizliliği (ürün ilkesi)**
12. Dissolüsyon verisi **loglanmaz**; yalnızca kullanıcı verisi loglanabilir — bu ilkeyi test/
    denetimle güvence altına al.

---

## Referanslar (APA)

Boddu, R., Kollipara, S., & Bhattiprolu, A. K. (2023). Dissolution profiles comparison using
conventional and bias corrected and accelerated f2 bootstrap approaches with different software's:
Impact of variability, sample size and number of bootstraps. *AAPS PharmSciTech, 25*(1).
https://doi.org/10.1208/s12249-023-02710-9

Costa, P., & Sousa Lobo, J. M. (2001). Modeling and comparison of dissolution profiles.
*European Journal of Pharmaceutical Sciences, 13*(2), 123–133.
https://doi.org/10.1016/s0928-0987(01)00095-1

Duan, J., Riviere, K., & Marroum, P. (2011). In vivo bioequivalence and in vitro similarity factor
(f2) for dissolution profile comparisons of extended release formulations: How and when do they
match? *Pharmaceutical Research, 28*(5), 1144–1156. https://doi.org/10.1007/s11095-011-0377-x

Endashaw, E., Tatiparthi, R., & Mohammed, T. (2022). Dissolution profile evaluation of seven brands
of amoxicillin-clavulanate potassium 625 mg tablets retailed in Hawassa Town, Sidama Regional State,
Ethiopia. *Research Square (preprint).* https://doi.org/10.21203/rs.3.rs-2233110/v1

Islam, M. M. (2018). Bootstrap confidence intervals for dissolution similarity factor f2.
*Biometrics & Biostatistics International Journal, 7*(5).
https://doi.org/10.15406/bbij.2018.07.00237

Moellenhoff, K., Dette, H., & Kotzagiorgis, E. (2018). Regulatory assessment of drug dissolution
profiles comparability via maximum deviation. *Statistics in Medicine, 37*(20), 2968–2981.
https://doi.org/10.1002/sim.7689

Romodanovsky, D. P., & Goryachev, D. V. (2021). Alternative methods for dissolution profile
comparison in the dissolution test. *Drug Development & Registration, 10*(4), 197–207.
https://doi.org/10.33380/2305-2066-2021-10-4-197-207

Sahoo, J., Murthy, P. N., & Biswal, S. (2008). Comparative study of propranolol hydrochloride
release from matrix tablets with Kollidon®SR or hydroxypropyl methylcellulose. *AAPS PharmSciTech,
9*(2), 577–582. https://doi.org/10.1208/s12249-008-9092-2

Samaha, D., Shehayeb, R., & Kyriacos, S. (2009). Modeling and comparison of dissolution profiles of
diltiazem modified-release formulations. *Dissolution Technologies, 16*(2), 41–46.
https://doi.org/10.14227/dt160209p41

Shah, V. P., Tsong, Y., & Sathe, P. (1998). In vitro dissolution profile comparison—Statistics and
analysis of the similarity factor, f2. *Pharmaceutical Research, 15*(6), 889–896.
https://doi.org/10.1023/a:1011976615750

Shah, V. P., Tsong, Y., & Sathe, P. (1999). Dissolution profile comparison using similarity factor,
f2. *Dissolution Technologies, 6*(3), 15. https://doi.org/10.14227/dt060399p15

Siswanto, A., Fudholi, A., & Nugroho, A. K. (2015). In vitro release modeling of aspirin floating
tablets using DDSolver. *Indonesian Journal of Pharmacy, 26*(2), 94.
https://doi.org/10.14499/indonesianjpharm26iss2pp94

Stevens, R. E., Gray, V. A., & Dorantes, A. (2015). Scientific and regulatory standards for
assessing product performance using the similarity factor, f2. *The AAPS Journal, 17*(2), 301–306.
https://doi.org/10.1208/s12248-015-9723-y

Wang, Y., Snee, R. D., & Keyvan, G. (2015). Statistical comparison of dissolution profiles.
*Drug Development and Industrial Pharmacy, 42*(5), 796–807.
https://doi.org/10.3109/03639045.2015.1078349

Zhai, S., Mathew, T., & Huang, Y. (2016). Comparison of drug dissolution profiles: A proposal based
on tolerance limits. *Statistics in Medicine, 35*(29), 5464–5476. https://doi.org/10.1002/sim.7072

Zhang, Y., Huo, M., & Zhou, J. (2010). DDSolver: An add-in program for modeling and comparison of
drug dissolution profiles. *The AAPS Journal, 12*(3), 263–271.
https://doi.org/10.1208/s12248-010-9185-1

Zuo, J., Gao, Y., & Bou-Chacra, N. A. (2014). Evaluation of the DDSolver software applications.
*BioMed Research International, 2014*, 204925. https://doi.org/10.1155/2014/204925

### Birincil regülasyon kaynakları (resmi kılavuzlar — scite dışı, doğrudan otorite belgeleri)
- FDA (1997). *Guidance for Industry: Dissolution Testing of Immediate Release Solid Oral Dosage
  Forms* (ve SUPAC-IR/MR kılavuzları).
- EMA/CHMP (Rev. 1). *Guideline on the Investigation of Bioequivalence* (CPMP/EWP/QWP/1401/98) —
  f2 geçerlilik koşulları ve alternatifler (Moellenhoff et al., 2018 içinde atıflanmıştır).
- USP <711> Dissolution monograf çerçevesi.

> Not: Yukarıdaki üç kılavuz birincil otorite belgeleridir; metindeki sayısal kurallar bu
> belgelere atıfla literatürden (özellikle Stevens et al., 2015; Moellenhoff et al., 2018)
> doğrulanmıştır.
