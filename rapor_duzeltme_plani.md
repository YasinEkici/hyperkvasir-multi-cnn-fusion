# Derin Öğrenme Dönem Projesi Raporu — Düzeltme ve Doğrulama Planı

Bu dosya, mevcut raporu ödev yönergesine göre daha doğal, akademik ve kapsam-uyumlu hale getirmek için hazırlanmış düzeltme planıdır. Amaç metni “tespit sistemlerini atlatmak” değil; raporu ödev gereksinimleriyle uyumlu, okunabilir, tutarlı ve savunulabilir hale getirmektir.

---

## 1. Doğrulama Özeti

### 1.1. Ödev kapsamı ile uyum durumu

Ödev yönergesinde beklenen ana unsurlar şunlardır:

- Üç farklı CNN modeli: **ResNet50, MobileNetV2, EfficientNetB0**
- Her CNN’den feature vector çıkarılması
- Özelliklerin en az iki yöntemle birleştirilmesi:
  - Concatenation
  - Weighted fusion
- Birleşik özellik vektörünün **MLP** ile sınıflandırılması
- Ablation study:
  - Tek CNN
  - İki CNN birleşimi
  - Üç CNN birleşimi
- Transfer learning karşılaştırması:
  - Sadece feature extraction / frozen özellik çıkarımı
  - Son 2 veya 3 blok fine-tuning
- Raporlanması gereken metrikler:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Rapor bölümleri:
  - Giriş
  - Yöntem
  - Deneyler
  - Sonuçlar
  - Tartışma

Mevcut rapor bu ana beklentileri büyük ölçüde karşılıyor. Raporun güçlü tarafı, yalnızca tek bir sonuç vermek yerine CNN omurgaları, füzyon yöntemleri ve transfer öğrenme stratejileri arasında kontrollü karşılaştırma yapmasıdır.

### 1.2. Sorun tipi

Tespit edilen sorunların çoğu bilimsel içerikten değil, raporun üslubundan ve bazı küçük tutarlılık problemlerinden kaynaklanıyor.

Ana sorunlar:

- Bazı ifadeler akademik rapordan çok “repo/agent iç notu” gibi duruyor.
- Bazı terimler gereğinden fazla savunmacı veya konuşma dili gibi.
- Bazı iç referanslar (`PLD-06`, `D-07`, `docs/...`, `results/tables/...`) final raporda fazla teknik/proje içi görünüyor.
- Küçük ama önemli sayısal ve tablo referansı tutarsızlıkları var.

---

## 2. Mutlaka Düzeltilmesi Gerekenler

### 2.1. “Donmuş final model” ifadesini değiştir

Mevcut kullanım:

```text
Donmuş final model
```

Sorun:

- Raporda zaten “frozen feature extraction” diye bir transfer learning rejimi var.
- “Donmuş final model” ifadesi hem teknik olarak kafa karıştırıyor hem de Türkçe akademik üslupta doğal durmuyor.

Önerilen kullanım:

```text
Nihai model
```


Örnek değişiklik:

```text
Nihai model olarak, exp 11 yapılandırmasının TTA uygulanmış sürümü seçilmiştir.
```

Bu değişiklik özellikle şu bölümlerde uygulanmalı:

- Özet
- 4.5 Toplamsal Ablasyonlar
- 4.6 Sınıf-Bazlı Çözümleme
- 4.8 Şekiller / Şekil 5 açıklaması
- 5.3 En İyi Kombinasyon
- Sonuç bölümü

---

### 2.2. İç proje/repo referanslarını ana metinden temizle

Aşağıdaki türden ifadeler final ödev raporunda fazla iç-doküman havası veriyor:

```text
PLD-06
PLD-11
PLD-18
D-06
D-07
D-08
D-09
docs/FINAL_MODEL.md
docs/environment.md
results/tables/...
src/models/fusion/...
finetune_wide.yaml
resolved_config.yaml
epoch_log.jsonl
```

Bu referanslar tamamen yanlış değil, ama ana metinde yoğun görünmeleri raporu “öğrenci raporu”ndan çok “agent/repo çıktısı” gibi gösteriyor.

Önerilen yaklaşım:

- Ana metinden `PLD-*`, `D-*`, `docs/...`, `results/...` referanslarını kaldır.
- Gerekirse şu tarz daha doğal bir cümle kullan:

```text
Deney konfigürasyonları, metrik çıktıları ve eğitim kayıtları proje dosyaları içinde saklanmıştır.
```

Örnek değişiklik:

```text
Proje kararları gereği sınıflandırıcı tüm deneylerde MLP olarak sabit tutulmuştur (PLD-06).
```

yerine:

```text
Bu çalışmada karşılaştırmayı CNN omurgaları, füzyon yöntemleri ve transfer öğrenme stratejileri üzerinde yoğunlaştırmak için sınıflandırıcı tüm deneylerde MLP olarak sabit tutulmuştur.
```

---

### 2.3. Sayısal tutarsızlık: MCC değerini düzelt

Raporda Tablo 4 içinde exp 11 için MCC değeri:

```text
0.8599
```

Tartışma bölümünde ise şu ifade geçiyor:

```text
MCC (0,8662)
```

Bu tutarsızlık düzeltilmeli.

Düzeltilecek cümle:

```text
MCC (0,8662) ise dengesizliğe nispeten dayanıklı tek bir skor olarak ek bir doğrulama sunar.
```

Önerilen hali:

```text
MCC (0,8599) ise dengesizliğe nispeten dayanıklı tek bir skor olarak ek bir doğrulama sunar.
```

Not: Yeni sayı uydurulmamalı. Tablo 4’teki değer hangisiyse tartışma bölümü onunla aynı yapılmalı.

---

### 2.4. Tablo referansı hatası: Tablo 3 → Tablo 6

3.3 Metrikler bölümünde sınıf-bazlı metrikler için şu tarz bir ifade var:

```text
sınıf başına kesinlik/duyarlılık/F1 ve destek değerleri (Tablo 3)
```

Fakat sınıf-bazlı metrikler raporda Tablo 6 olarak veriliyor.

Düzeltme:

```text
sınıf başına kesinlik/duyarlılık/F1 ve destek değerleri (Tablo 6)
```

Bu küçük bir hata ama rapor kalitesi açısından önemli.

---

### 2.5. Savunmacı veya konuşma dili gibi duran ifadeleri akademikleştir

Aşağıdaki ifadeler metni gereğinden fazla savunmacı veya yapay gösteriyor:

```text
dürüstçe raporlanmalıdır
gizlenmemeli
CE şampiyonu
ikinci tohum daha zayıf çekiliştir
mimarinin bu protokoldeki tavanına yakın olduğunu gösterir
yasaktır
manşet sonuç
```

Önerilen akademik karşılıklar:

| Mevcut ifade | Önerilen ifade |
|---|---|
| Bu nötr sonuç gizlenmemeli, dürüstçe raporlanmalıdır. | Bu sonuç, GMU’nun bu veri kümesi ve protokol altında ek bir kazanım sağlamadığını göstermektedir. |
| CE şampiyonu | temel çapraz-entropi yapılandırması |
| ikinci tohum daha zayıf çekiliştir | ek tohumlarla elde edilen topluluk makro-F1 değerini artırmamıştır |
| mimarinin bu protokoldeki tavanına yakın olduğunu gösterir | bu ek tekniklerle elde edilebilecek iyileşmenin sınırlı kaldığını göstermektedir |
| yasaktır | bu çalışmada kullanılmamıştır / tercih edilmemiştir |
| manşet sonuç | ana sonuç / raporlanan temel sonuç |

---

### 2.6. “3 sınıflandırıcı” meselesini daha sakin açıkla

Ödev yönergesinde bir yerde “3 sınıflandırıcı bazlı” ifadesi geçiyor; ancak gereksinimler bölümünde sınıflandırıcı olarak MLP belirtilmiş.

Mevcut rapordaki ifade biraz savunmacı duruyor:

```text
Ödev metnindeki “3 sınıflandırıcı” ifadesinin sehven yazıldığı eğitmen tarafından teyit edilmiştir...
```

Daha doğal öneri:

```text
Ödev yönergesinde sınıflandırıcı olarak MLP belirtildiğinden, bu çalışmada sınıflandırıcı sabit tutulmuş; karşılaştırmalar CNN omurgaları, füzyon yöntemleri ve transfer öğrenme stratejileri üzerinden yapılmıştır.
```

Buradan eğitmen kısmını komple çıkar, çünkü başından beri sadece MLP olduğunu biliyorduk.

---

### 2.7. A100 / provenance / git SHA bölümünü kısalt

Mevcut 3.5 Yeniden Üretilebilirlik ve Ortam bölümü gereğinden fazla MLOps/repo detayı içeriyor.

Aşağıdaki türden detaylar ödev raporu için fazla ağır duruyor:

```text
Colab A100 yolu
provenance kapısı
git SHA
resolved_config.yaml
veri kümesi SHA256
manifest kimliği
orchestration
```

Önerilen kısa versiyon:

```text
Deneyler yerel RTX 5080 GPU ortamında yürütülmüştür. Yeniden üretilebilirlik için sabit tohum kullanılmış, deney konfigürasyonları ve metrik çıktıları proje dosyaları içinde saklanmıştır. Raporlanan 5-katlı çapraz doğrulama ve final değerlendirme sonuçları aynı veri bölme protokolü altında üretilmiştir.
```

Bu hali yeterince teknik ama final ödev raporuna daha uygun.

---
---

## 4. Uygulanacak Düzeltme Sırası

### Aşama 1 — Kritik tutarlılık düzeltmeleri

1. MCC değerini Tablo 4 ile uyumlu hale getir.
2. Tablo 3 / Tablo 6 referans hatasını düzelt.
3. “Donmuş final model” ifadesini “nihai model” veya “sabitlenen final model” olarak değiştir.
4. “CE şampiyonu”, “dürüstçe”, “gizlenmemeli”, “çekiliş”, “manşet sonuç”, “yasaktır” gibi ifadeleri akademik dille değiştir.

### Aşama 2 — Repo/agent izlerini temizleme

1. Ana metinden `PLD-*`, `D-*`, `docs/...`, `results/tables/...`, `src/...` gibi iç referansları kaldır.
2. “Kaynak: results/tables/...” geçen tablo açıklamalarını sadeleştir.
3. Gerekirse bu kaynak bilgilerini rapor içinde değil, kod teslimindeki dosya yapısında bırak.

Önerilen tablo açıklaması:

```text
Tablo 2: Seçilmiş 5-katlı CV sonuçları, ortalama ± standart sapma.
```

yerine:

```text
Tablo 2: Seçilmiş 5-katlı çapraz doğrulama sonuçları, ortalama ± standart sapma.
```

### Aşama 3 — Bölüm sadeleştirmesi

1. 4.5 Toplamsal Ablasyonlar bölümünde “dürüst baş-başa” gibi ifadeleri değiştir.

Önerilen başlık:

```text
4.5 Toplamsal Ablasyonlar
```

veya:

```text
4.5 Ek Ablasyon Deneyleri
```

### Aşama 4 — Son okuma

Aşağıdaki terimler için raporda arama yap:

```text
Donmuş
PLD-
D-
docs/
results/
src/
dürüst
gizlenmemeli
şampiyon
çekiliş
yasaktır
manşet
provenance
orchestration
0,8662
Tablo 3
```

Her eşleşmeyi tek tek kontrol et.

---

## 5. Bilimsel Sonuçlar Korunmalı

Düzeltmeler sırasında aşağıdaki bilimsel kararlar korunmalı:

- Ana metrik makro-F1 olarak kalmalı.
- Accuracy destekleyici metrik olarak kalmalı.
- ResNet50, MobileNetV2 ve EfficientNetB0 karşılaştırması korunmalı.
- Concatenation ve weighted fusion zorunlu yöntemler olarak açık kalmalı.
- MLP sınıflandırıcı olarak sabit tutulmalı.
- Frozen feature extraction ve fine-tuning karşılaştırması korunmalı.
- GMU, focal loss, TTA ve ensemble ek ablasyon olarak kalabilir.
- “State-of-the-art” veya doğrudan üstünlük iddiası eklenmemeli.
- Protokol farkları nedeniyle literatür sonuçlarının bağlamsal olduğu vurgusu korunmalı.
- Yeni deney sonucu veya yeni sayı uydurulmamalı.

---

---

## 6. Beklenen Sonuç

Bu plan uygulandıktan sonra rapor:

- Ödev kapsamıyla daha net uyumlu görünecek.
- Gereksiz “agent/repo iç dokümanı” havasından arınacak.
- Daha doğal ve akademik bir Türkçe üsluba yaklaşacak.
- Sayısal ve tablo referansı hataları düzelecek.

