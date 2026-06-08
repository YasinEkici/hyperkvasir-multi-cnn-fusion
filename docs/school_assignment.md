**3.5.2026**

**DERİN ÖĞRENME DERSİ**

**Çoklu CNN Tabanlı Özellik Füzyonu ile Sınıflandırma**

- **1\. Amaç**

Bu ödevin amacı, öğrencilerin:

- Evrişimsel Sinir Ağları (CNN) ile özellik çıkarımı (feature extraction),
- Farklı mimarilerin öğrenmiş olduğu temsillerin karşılaştırılması,
- Özellik füzyonu (feature fusion) tekniklerinin uygulanması,
- Elde edilen özelliklerin bir sınıflandırıcı ile değerlendirilmesi

konularında uygulamalı deneyim kazanmasını sağlamaktır.

- **2\. Problem Tanımı**

Bu ödev kapsamında, bir görüntü veri seti üzerinde:

- **Üç farklı CNN modeli** tasarlamanız / kullanmanız,
- Her modelden **özellik (feature) vektörü çıkarmanız**,
- Bu özellikleri birleştirerek (**feature fusion**) tek bir temsil oluşturmanız,
- Bu birleşik temsili kullanarak bir **sınıflandırıcı modeli eğitmeniz**

beklenmektedir.

- **3\. Gereksinimler**

**3.0 Veri Seti**

Veri kümesi her grup farklı bir çalışmada kullanılan public erişilebilen, 2 sınıftan fazla sınıf etiketi bulunan bir sınıflandırma veri kümesi seçecektir. Bu veri kümesi gruplara bırakılmış olup herhangi bir sınıflandırma işlemi seçilebilir. Seçilen veri kümesinin raporda son 3 yılda en az 2-3 çalışmada kullanılmış olması projenin başarının diğer projelerle karşılaştırması için önem taşımaktadır. Veri setleri UCI, Kaggle gibi bilinen repository'lerden elde edilmeli ve referansları raporda verilmelidir.

**3.1 CNN Modelleri**

Aşağıdaki kriterlere uygun **üç farklı CNN modeli** kullanılmalıdır:

- Resnet50
- MobileNetV2
- EfficientNetB0

**3.2 Özellik Çıkarma**

Her CNN modeli için:

- Son sınıflandırma katmanı kaldırılmalı,
- Ara katmandan veya son özellik katmanından **feature vector** elde edilmelidir.

**3.3 Özellik Füzyonu**

Elde edilen üç farklı özellik vektörü:

- Aşağıdaki klasik birleştirme ve ağırlıklı birleştirme ile 2 yöntemle birleştirilmesi gerekmektedir.

Zorunlu yöntemler:

- **Concatenation (klasik birleştirme)**
- **Weighted fusion (ağırlıklı birleştirme)**

**3.4 Sınıflandırıcı**

Birleşik özellik vektörü aşağıdaki yöntemle sınıflandırın.

- Çok Katmanlı Yapay Sinir Ağı (Fully Connected Neural Network (MLP))
- **4\. Deneysel Çalışmalar (Zorunlu)**

**4.1 Karşılaştırmalı Analiz (Ablation Study)**

Aşağıdaki senaryolar mutlaka test edilmelidir:

- Tek CNN modeli ile sınıflandırma
- İki CNN'in birleşimi
- Üç CNN'in birleşimi

Her birinin performansı sınıflandırıcı bazında da ayrı ayrı karşılaştırılmalıdır.

**4.2 Transfer Learning Aşaması**

Aşağıdaki 2 yaklaşım kullanılmalıdır

- Sadece Feature Extraction ile gerçekleştirin (özelliklerin hepsi hazır modelden elde edilsin ve sadece sınıflandırma aşamasında öğrenme olsun)
- Son 2 veya 3 konvolüsyon+pooling katmanındaki ağırlıkları güncelleyerek Fine- Tuning (İnce Ayar) yapın.

Her birinin performansı sınıflandırıcı bazında da ayrı ayrı karşılaştırılmalıdır.

**4.2 Performans Metrikleri**

Aşağıdaki metrikler raporlanmalıdır:

- Accuracy
- Precision
- Recall
- F1-score
- **5\. Rapor İçeriği**

Rapor aşağıdaki bölümleri içermelidir:

**5.1 Giriş**

- Problem tanımı
- Kullanılan yaklaşımın genel açıklaması
- Karşılaştırmalar
  - Birleştirme yaklaşımı bazlı
  - CNN modeli bazlı
  - 3 sınıflandırıcı bazlı yapılmalıdır

**5.2 Yöntem**

Karşılaştırma yapılabilecek seçeneklerin hepsi:

- CNN mimarileri
- Feature extraction yöntemi
- Fusion yöntemi
- Transfer Learning Yöntemi

**5.3 Deneyler**

- Veri seti açıklaması
- Eğitim detayları
- Hiperparametreler

**5.4 Sonuçlar**

- Tablolar ve grafikler
- Karşılaştırmalar

**5.5 Tartışma (EN KRİTİK BÖLÜM)**

Aşağıdaki sorulara cevap verilmelidir:

- Hangi model daha iyi özellik öğrenmiştir? Neden?
- Feature fusion performansı neden artırmış veya artırmamış olabilir?
- Hangi kombinasyon en iyi sonucu vermiştir?
- Modellerin güçlü ve zayıf yönleri nelerdir?
- Hangi transfer learning yaklaşımı daha başarılı ve ne kadar sürmüş?
- **6\. Teslim**

Teslim edilmesi gerekenler:

- Kaynak kod (Python, PyTorch / TensorFlow)
- Rapor (PDF formatında)
- (Opsiyonel) Eğitim logları ve grafikler
- **7\. Değerlendirme Kriterleri**

| **Kriter**               | **Yüzde** |
| ------------------------ | --------- |
| Model tasarımı           | %15       |
| Doğru implementasyon     | %20       |
| Deneysel analiz          | %20       |
| Rapor kalitesi           | %20       |
| Yorum, tartışma ve sunum | %25       |

**8\. Akademik Dürüstlük**

- Kodun tamamı size ait olmalıdır.
- Hazır kodlar kullanılabilir ancak mutlaka referans verilmelidir.
- Anlamadan yapılan çözümler rapor kısmında açıkça anlaşılacaktır.

**9\. Notlar**

- Amaç sadece çalışan bir model üretmek değil, **anlamlı analiz yapmaktır**.
- Basit çözümler kabul edilir, yüzeysel yorumlar kabul edilmez.
- Yaratıcı yaklaşımlar ekstra puan getirebilir.

**ÖNEMLİ:** Yaptığınız projenin tüm adımlarının raporlanması beklenmektedir. Ekampüse hem kodlarınız hem de ödev raporunuzu eklemeniz gerekmektedir. Tüm dosyaları (projenin raporu, kodları ve sunumunuzu) " birinci*ogrenci_adi_numarasi* ikinci*ogrenci_adi_numarasi*.rar" olarak yüklemeniz gerekmektedir. Ödev sunumlarınız kalan haftalar yetişmeyeceği için mutlaka Youtube üzerine koyduğunuz videolara göre değerlendireceğim. Youtube'a video eklemeyen ama ekampüse rapor ve kodlarını gönderen öğrencilerin projesi değerlendirilemeyecektir. Youtube'a video yükleme ve ekampüse ilgili "rar" dosyasını son yükleme tarihi 8 Haziran 2026 perşembe saat 23:59'a kadardır. Süre az kaldığı için bu tarihten sonra yüklenen ödevler ve videolar kabul edilmeyecektir. Öğrencinin ödevden not alabilmesi için video yüklemesi ve ekampüs'e proje raporu ve kodları içeren sıkıştırılmış dosyayı yüklemesi gerekmektedir. Başarılar dilerim.
