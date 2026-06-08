# Sunum + Video Anlatım Metni

> **Amaç:** Bu dosya hem slayt destesini hem YouTube video anlatımını besler.
> **Hedef süre:** ~8–12 dk. **Dil:** Türkçe. **Kural:** yalnızca belgelenen sayılar;
> headline metrik macro-F1, accuracy destekleyici; SOTA iddiası YOK; protokol farklı
> literatür yalnızca bağlamsal. Her slayttaki konuşma metni doğal/öğrenci üslubudur.
>
> Slaytlardaki sayılar `docs/FINAL_MODEL.md` ve `04_results.tex` tablolarına izlenebilir.

---

## Slayt 1 — Başlık (~20 sn)
**İçerik (slayt):**
- Başlık: "Çoklu CNN Tabanlı Özellik Füzyonu ile HiperKvasir 23-Sınıf GI Görüntü Sınıflandırması"
- Öğrenci ad–soyad–no (×2)
- Ders / tarih

**Görsel:** —
**Konuşma:** "Merhaba. Bu sunumda, HiperKvasir veri kümesindeki 23 sınıflı gastrointestinal endoskopi görüntülerini, üç farklı CNN omurgasının özelliklerini birleştirerek sınıflandıran projemizi anlatacağım. Önce problemi ve veriyi, sonra yöntemi, sonuçları ve dürüst bir değerlendirmeyi paylaşacağım."

---

## Slayt 2 — Problem ve Veri Seti (~60 sn)
**İçerik:**
- HiperKvasir etiketli alt küme: 23 sınıf, 10.662 görüntü (CC BY 4.0)
- Ciddi sınıf dengesizliği: polyp ~1028 örnek ↔ hemorroids 6, ileum 9
- Bu yüzden tek başına **doğruluk yanıltıcı**; ana ölçüt **macro-F1** (her sınıfa eşit ağırlık)
- Üç kontrollü karşılaştırma ekseni: omurga / füzyon / transfer öğrenme

**Görsel:** (opsiyonel) birkaç örnek sınıf görseli ya da sınıf dağılımı çubuğu
**Konuşma:** "Veri kümesinde 23 sınıf ve toplam 10.662 görüntü var, ama dağılım son derece dengesiz: bazı sınıfta binin üzerinde örnek varken, hemorroids'te yalnızca 6 örnek var. Böyle bir veride yüksek genel doğruluk, nadir sınıflardaki başarısızlığı gizleyebilir. Bu yüzden başarıyı macro-F1 ile ölçüyoruz; her sınıfa eşit ağırlık verdiği için gerçek davranışı gösterir. Çalışmayı üç eksende karşılaştırıyoruz: omurga ağı, füzyon yöntemi ve transfer öğrenme."

---

## Slayt 3 — Yöntem / Mimari (~75 sn)
**İçerik:**
- 3 önceden eğitilmiş omurga: ResNet50 (2048-d), MobileNetV2 (1280-d), EfficientNetB0 (1280-d)
- Her dal → 512-d projeksiyon (Linear→LayerNorm→GELU)
- Füzyon: concat / weighted / GMU
- Sınıflandırıcı: **yalnızca MLP** (256 + dropout) — proje kararı (PLD-06)
- Not: ödevdeki "3 sınıflandırıcı" ifadesi eğitmence hatalı teyit edildi → sınıflandırıcı sabit MLP

**Görsel:** `architecture.pdf`
**Konuşma:** "Mimari şöyle işliyor: üç önceden eğitilmiş CNN omurgası özellik çıkarıcı olarak kullanılıyor; çıkış boyutları farklı olduğu için her dal 512 boyutlu ortak bir uzaya izdüşürülüyor. Ardından bu özellikler üç yöntemden biriyle birleştiriliyor — basit birleştirme, öğrenilebilir ağırlıklı füzyon, ve gelişmiş kapılı füzyon GMU. Son temsili tek bir MLP sınıflandırıyor. Sınıflandırıcı tüm deneylerde MLP olarak sabit; ödevdeki '3 sınıflandırıcı' ifadesinin sehven yazıldığı eğitmen tarafından teyit edildi."

---

## Slayt 4 — Deneysel Kurulum (~55 sn)
**İçerik:**
- Resmi HiperKvasir 5-katlı (5-fold) protokol, sızıntı denetimli
- Transfer öğrenme: frozen özellik çıkarımı vs son 3 blok fine-tuning (BN dondurulmuş)
- Metrikler: macro-F1 (ana), accuracy/weighted-F1/MCC (destek)
- Güvenilirlik: 5-kat mean±std + bootstrap %95 güven aralığı (n=10.662)

**Görsel:** (opsiyonel) kısa hiperparametre kutusu
**Konuşma:** "Tüm karşılaştırmaları resmi 5-katlı bölme üzerinde yaptık; bu, yapılandırmaların aynı veri bölmesinde adil kıyaslanmasını sağlıyor. İki transfer öğrenme yaklaşımını karşılaştırıyoruz: omurgaların donduğu frozen çıkarım ve son üç bloğun ince ayarlandığı fine-tuning. Sonuçları sadece ortalama değil, bootstrap %95 güven aralıklarıyla da raporluyoruz; böylece farkların istatistiksel olarak anlamlı olup olmadığını görebiliyoruz."

---

## Slayt 5 — Sonuçlar: Karşılaştırma (omurga / füzyon / transfer) (~80 sn) [§5.5]
**İçerik:**
- Frozen'da en iyi tekil omurga: **EfficientNetB0** (macro-F1 0.5586)
- En iyi CV yapılandırması: **üçlü ağırlıklı fine-tune (exp 11)** — F1 0.5892±0.0102, pooled 0.6000
- Ağırlıklı füzyon, birleştirmeden **istatistiksel olarak ayırt edilebilir** (CI'ler örtüşmüyor: 0.5814 > 0.5799)
- GMU nötr (Δ≈0.002, CI'ler örtüşür)
- Fine-tune, CV ortalamasında frozen'ı geçer

**Görsel:** `comparison_bar_chart.pdf` + Tablo 2 (CV) + Tablo 2b (CI)
**Konuşma:** "Tekil omurgalar arasında EfficientNetB0 frozen rejimde en iyisi. Ama asıl kazanan füzyon: beş katın tamamında en iyi yapılandırma üçlü ağırlıklı fine-tune, macro-F1 0.59. Önemli nokta şu — ağırlıklı füzyon, basit birleştirmeden istatistiksel olarak ayırt edilebiliyor; güven aralıkları örtüşmüyor. Buna karşılık gelişmiş GMU, basit birleştirmeyi geçemiyor; aradaki fark gürültü içinde. Transfer öğrenme tarafında ise fine-tuning, çapraz doğrulama ortalamasında frozen çıkarımı tutarlı biçimde geçiyor."

---

## Slayt 6 — Week 3.5 Ablasyonları: Dürüst Değerlendirme (~75 sn)
**İçerik:**
- En iyi config üzerinde dört ek teknik denendi (havuzlanmış macro-F1):
  - CE şampiyon (exp 11): **0.6000** [0.5814, 0.6206]
  - Focal loss: 0.5914 (altında)
  - Seed ensemble: 0.5971
  - **TTA: 0.6075** [0.5860, 0.6296] — en iyi nokta tahmini
- **Hiçbiri CE'yi %95 CI düzeyinde geçemedi** (tüm aralıklar örtüşüyor)
- Final model = **exp 11 + TTA** (sıfır ek eğitim maliyeti, en iyi nokta tahmini)

**Görsel:** Tablo (Week 3.5 baş-başa) — `04_results.tex` `tab:week35`
**Konuşma:** "En iyi yapılandırmanın üzerine dört ek teknik denedik: focal loss, TTA, ve sızıntısız seed topluluğu. Dürüst sonuç şu: hiçbiri temel çapraz-entropi modelini %95 güven aralığı düzeyinde geçemedi — tüm aralıklar örtüşüyor. Focal nokta olarak altta kaldı; TTA en iyi nokta tahminini verdi ama fark istatistiksel olarak anlamlı değil; topluluk macro-F1'i artırmadı. Bu, mimarinin bu protokolde tavanına yakın olduğunu gösteriyor. Final modeli, sıfır ek eğitim maliyetiyle en iyi nokta tahminini veren exp 11 + TTA olarak dondurduk."

---

## Slayt 7 — Hata Analizi: Nadir Sınıflar (~70 sn) [§5.5]
**İçerik:**
- macro-F1 (~0.60) ↔ accuracy (~0.87) arasında ~0.27 fark = dengesizliğin doğrudan kanıtı
- Yüksek destekli sınıflar güçlü: retroflex-stomach 0.99, bbps-2-3 0.97, normal-pylorus 0.98
- Çok düşük destekli nadir sınıflar çöküyor: ileum (9) ve uc-grade-1-2 (11) F1≈0; hemorroids (6) 0.25
- Hata matrisi: nadir sınıf, görsel/anlamsal olarak en yakın çoğunluk sınıfına soğuruluyor

**Görsel:** `confusion_matrix_frozen_tta.pdf` + `per_class_f1_frozen_tta.pdf` (+ Tablo 3)
**Konuşma:** "Dengesizliğin etkisi sayısal olarak çok net: doğruluk yaklaşık 0.87 iken macro-F1 yaklaşık 0.60. Bu 0.27'lik fark, az sayıdaki kalabalık sınıfın metrikleri yukarı çektiğini gösteriyor. Çok destekli sınıflarda model 0.95'in üzerinde F1 alıyor; ama ileum veya ulseratif kolit ara dereceleri gibi sadece birkaç örnekli sınıflarda F1 sıfıra düşüyor. Hata matrisine bakınca bunun rastgele olmadığını görüyoruz: nadir sınıflar, görsel olarak en benzedikleri çoğunluk sınıfına karışıyor. Yani bu bir model hatası değil, veri kıtlığı sınırı."

---

## Slayt 8 — Tartışma: §5.5 Soruları (~75 sn) [§5.5]
**İçerik (kısa yanıtlar):**
- Hangi omurga iyi öğrendi? Tekil: EfficientNetB0; ama üçlü ağırlıklı füzyon hepsini geçiyor
- Füzyon yardımcı oldu mu? Evet (ağırlıklı, CI-ayrık); GMU ek katkı vermedi
- En iyi kombinasyon: üçlü ağırlıklı fine-tune + TTA
- Fold-0 yanılgısı: fold-0'da concat>weighted; 5-fold tersine çevirdi (seçim yanlılığından kaçındık)
- Frozen vs fine-tune + süre: fine-tune daha iyi ama ~18–21 dk/kat; frozen önbellekle çok hızlı

**Görsel:** (madde listesi slaytı)
**Konuşma:** "Tartışmayı ödevin sorularıyla özetleyeyim. En iyi tekil omurga EfficientNetB0, ama tümleyici bilgiyi birleştiren üçlü ağırlıklı füzyon daha ayırt edici. Füzyon işe yaradı; ağırlıklı füzyon basit birleştirmeden anlamlı biçimde iyi, GMU ise ek katkı sağlamadı. Önemli bir metodolojik ders: tek bir folda birleştirme öne çıkıyordu, ama 5-katlı doğrulama bunu tersine çevirdi — tek folda karar vermek seçim yanlılığı olurdu. Maliyet açısından fine-tune kat başına yaklaşık 18–21 dakika sürerken frozen, önbelleğe alınmış özelliklerle saniyeler mertebesinde."

---

## Slayt 9 — Literatürle Bağlamsal Konumlandırma (~55 sn)
**İçerik:**
- Tek protokol-uyumlu çapa: Borgli resmi-bölme **macro-F1 ≈ 0.62** (bizim ≈ 0.60 aynı bantta)
- Effimix ≈ %98 doğruluk, GastroViT ≈ %92 doğruluk / 0.64 F1 → **farklı protokol → bağlamsal**
- Doğrudan üstünlük / SOTA iddiası YOK; GastroViT 16-sınıf sonuçları bizimkiyle karıştırılmaz

**Görsel:** (madde listesi / küçük bağlamsal tablo)
**Konuşma:** "Sonuçlarımızı literatüre dürüstçe yerleştirelim. Tek protokol-uyumlu kıyas Borgli'nin resmi-bölme sonucu, macro-F1 yaklaşık 0.62; bizim 0.60'ımız aynı bantta. Effimix ve GastroViT çok daha yüksek doğruluk bildiriyor, ama bunlar farklı bölme ve ön işleme kullanıyor, dolayısıyla yalnızca bağlamsal olarak anılabilir; doğrudan üstünlük ya da güncel-en-iyi iddiasında bulunmuyoruz."

---

## Slayt 10 — Sonuç, Sınırlamalar, Gelecek Çalışma (~50 sn)
**İçerik:**
- Final model: üçlü ağırlıklı CE fine-tune + TTA, macro-F1 0.6075 [0.5860, 0.6296]
- Güçlü yön: resmi 5-fold + bootstrap CI ile metodolojik olarak tutarlı, dürüst değerlendirme
- Sınırlamalar: nadir-sınıf veri kıtlığı; protokol-farklı literatüre doğrudan kıyas yok; tek tohum
- Gelecek: Grad-CAM++/UMAP yorumlanabilirlik; nadir sınıflar için veri-merkezli yaklaşımlar

**Görsel:** (özet madde slaytı)
**Konuşma:** "Özetle, final modelimiz üçlü ağırlıklı fine-tune'un TTA'lı sürümü; havuzlanmış macro-F1 0.6075. Çalışmanın en güçlü yanı, her şeyin resmi protokol ve güven aralıklarıyla, abartısız ve dürüst biçimde raporlanması. Başlıca sınırlama nadir sınıflardaki veri kıtlığı. Gelecekte Grad-CAM++ ve UMAP ile yorumlanabilirlik ve nadir sınıfları hedefleyen veri-merkezli yöntemler değerlendirilebilir."

---

## Slayt 11 — Teşekkür / Referanslar (~20 sn)
**İçerik:**
- Teşekkür
- Anahtar referanslar: Borgli 2020 (HiperKvasir), Arevalo 2017 (GMU), Lin 2017 (Focal), He 2016 / Sandler 2018 / Tan 2019 (omurgalar)

**Görsel:** —
**Konuşma:** "Dinlediğiniz için teşekkürler. Kullandığımız temel kaynaklar veri kümesi için Borgli ve arkadaşları, füzyon ve yöntem için ilgili çalışmalar. Sorularınızı yanıtlamaktan memnuniyet duyarım."

---

## Kayıt Notları
- **Ekran kaydı:** PowerPoint "Kaydet" veya OBS Studio yeterli; 1080p, slayt + sesli anlatım.
- **Süre:** her slaytın yanındaki tahmini süreyi tut; **toplam ≤ 12 dk** kalmaya çalış.
- **Akış:** slayt 1–4 ~3,5 dk (problem+yöntem), 5–8 ~5 dk (sonuç+tartışma), 9–11 ~2 dk (literatür+kapanış).
- **Figürler:** doğrudan rapordaki PDF figürlerini kullan (architecture, comparison_bar_chart, confusion_matrix, per_class_f1) — tutarlılık için.
- **Dürüstlük:** "hiçbir ek teknik CE'yi CI düzeyinde geçmedi" ve "SOTA iddiası yok" cümlelerini mutlaka söyle (rapordaki çerçeveyle birebir).
- **Yükleme:** YouTube'a "unlisted" (liste dışı) yükle, linki `.rar` ve rapora ekle.
