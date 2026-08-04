# jacopy

Dereceli cebir, bracket kalkülüsü ve diferansiyel geometri için sembolik
hesaplama kütüphanesi. Ayırt edici özelliği: her manipülasyon **adım adım
izlenebilir bir kanıt zinciri** (`ProofChain`) olarak üretilir — sonuçlar
kara kutudan çıkmaz, her adımın hangi tanım/aksiyom/teoremle atıldığı
kayıtlıdır ve bir özdeşlik kapanmadığında kalan engel (kalıntı) açıkça
raporlanır.

Kapsam: Cartan kalkülüsü (d, L, ι), Lie / Courant-ailesi algebroid'leri
(beyan sistemi), Schouten-Nijenhuis, Poisson / symplectic / Nambu-Poisson
geometrisi, Koszul bracket'leri, metric-affine geometri (torsiyon,
eğrilik, Bianchi, Cartan yapı denklemleri), bialgebroid kalkülüsü.

Saf Python'dur; çalışmak için hiçbir dış pakete ihtiyaç duymaz.

---

## Kurulum — adım adım (hiç kurulum yapmamış biri için)

### 1. Python'u kurun (3.10 veya üstü)

- **macOS / Linux**: Terminal'i açıp şunu yazın; `3.10` veya daha
  büyük bir sürüm görüyorsanız bu adımı atlayın:

  ```bash
  python3 --version
  ```

- Kurulu değilse [python.org/downloads](https://www.python.org/downloads/)
  adresinden indirin (Windows'ta kurulum sırasında **"Add Python to
  PATH"** kutusunu işaretleyin).

### 2. Projeyi indirin

Terminal'de (Windows'ta "PowerShell"):

```bash
git clone https://github.com/oselcukk/JacoPy.v.3.git
cd JacoPy.v.3.
```

> `git` yoksa: [git-scm.com/downloads](https://git-scm.com/downloads)
> adresinden kurun, ya da GitHub sayfasındaki yeşil **Code ▸ Download
> ZIP** düğmesiyle indirip klasöre girin.

### 3. Sanal ortam oluşturun ve etkinleştirin

Sanal ortam, projenin kendi izole Python alanıdır — sisteminize hiçbir
şey bulaştırmaz.

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# Windows PowerShell'de bunun yerine:  .venv\Scripts\Activate.ps1
```

Satır başında `(.venv)` görüyorsanız hazırsınız.

### 4. Paketi kurun

```bash
pip install -e ".[test]"
```

### 5. Testleri çalıştırıp doğrulayın

```bash
pytest -q
```

Sonunda `passed` yazan yeşil bir satır görmelisiniz (800+ test).
Görüyorsanız kurulum tamamdır.

### 6. İlk kanıtınız

Aşağıdakini `ilk_kanit.py` adıyla kaydedin ve `python ilk_kanit.py`
ile çalıştırın:

```python
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import functions, vector_fields
from jacopy.central.tangent.cartan import prove_cartan_magic_on_functions

reg = PropertyRegistry()
(f,) = functions("f", registry=reg)
(X,) = vector_fields("X")

# Cartan'ın sihirli formülü fonksiyonlarda: L_X f = ι_X(df) + d(ι_X f)
chain = prove_cartan_magic_on_functions(X, f, registry=reg)

print(f"Kanıt {len(chain.steps)} adımda kapandı:")
for step in chain.steps:
    print(f"  [{step.provenance_tag}] {step.rule}")
```

Çıktıda her adımın hangi kuralla atıldığını göreceksiniz — `axiom`
etiketi tanımsal kuralları, `theorem` etiketi kanıtlanmış sonuçların
alıntılanmasını gösterir.

---

## Hızlı tur

```python
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import Bundle, forms, functions, vector_fields

reg = PropertyRegistry()
f, g, h = functions("f g h", registry=reg)     # skaler fonksiyonlar
X, Y = vector_fields("X Y")                     # vektör alanları
(omega,) = forms("ω", degree=2)                 # 2-form

# Poisson geometrisi — Jacobi, [π,π]=0 beyanı altında kanıtlanır:
from jacopy.packages.poisson import poisson_structure, prove_poisson_jacobi
P = poisson_structure()
chain, cited = prove_poisson_jacobi(P, f, g, h, registry=reg)

# Soyut algebroid + beyan sistemi:
from jacopy.central.algebroid import algebroid
E = algebroid("E", Bundle("E"), declare=("local", "regular", "jacobi"))
u, v = E.sections("u v")
```

Bir kanıt kapanmazsa `ProofFailure` fırlatılır ve **kalıntı**
(kapatılamayan terimler) mesajda görünür — çoğu zaman bu kalıntı, hangi
aksiyomu beyan etmeniz gerektiğini söyler.

## Mimari

```
jacopy/
├── core/         # ifade ağacı: Expr, Sum, Product, Wedge, MultiEval, degree
├── algorithms/   # simplify, product_rule (graded Leibniz), distribute, ...
├── algebra/      # Act, Derivation, komütatörler, Lie bracket
├── brackets/     # GradedBracket soyutlaması
├── proof/        # kanıt motoru: ProofChain, ExpansionEngine, TheoremBook
├── central/      # hem TM hem algebroid E üzerinde ortak katman
│   ├── objects/  #   fonksiyon, vektör alanı, form, multivektör, frame, metrik
│   ├── tangent/  #   TM özel hâli: Lie bracket, d/L/ι, Schouten-Nijenhuis
│   └── algebroid/#   beyan sistemi (leibniz/lie/courant seviyeleri), locality
└── packages/     # alan paketleri
    ├── metric_affine/  # konneksiyon, T/R/Q, Bianchi, Koszul, E-versiyonlar
    ├── poisson/        # Poisson, symplectic, Koszul bracket, tilde, Nambu
    └── drinfeld/       # bialgebroid kalkülüsü (geliştiriliyor)
```

**Central-code ilkesi:** her nesne öyle tanımlanır ki `E = TM`,
`ρ_E = id`, `[·,·]_E = [·,·]_Lie` alındığında algebroid versiyonu her
zamanki (usual) versiyona indirgenir.

## Tasarım ilkeleri

1. **Tek kanonik tanım** — her operatörün bir tanımı vardır; geri kalan
   her şey teoremdir ve mekanik olarak türetilir.
2. **Türetilebilir özellikler asla sessizce varsayılmaz** — aksiyomlar
   (Jacobi, antisimetri, metrik-invarians, FI, dω=0, ...) açık
   **beyanlarla** girer; beyansız denemeler dürüstçe başarısız olur.
3. **Her adım izlenebilir** — zincirde her adımın kaynağı görünür:
   `axiom` (tanımsal kural), `theorem` (kanıtlanmış sonucun alıntısı)
   ya da açıkça etiketlenmiş sentetik çıkarım.

## Lisans

Kaynağı açık, tescilli lisans — ayrıntılar için [LICENSE](LICENSE)
dosyasına bakın.
