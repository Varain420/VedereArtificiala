# Laborator 1 - Îmbunătățirea imaginilor


import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request


# --- Funcție ajutătoare pentru a descărca și încărca o imagine de test ---
def incarcare_imagine_test(url, nume_fisier='imagine_test.jpg'):
    """Descarcă o imagine de la un URL și o încarcă folosind OpenCV."""
    try:
        urllib.request.urlretrieve(url, nume_fisier)
        img = cv2.imread(nume_fisier)
        if img is None:
            raise Exception("Imaginea nu a putut fi încărcată. Verificați calea sau formatul fișierului.")
        # OpenCV încarcă imaginile în format BGR, le convertim în RGB pentru afișare corectă cu Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Eroare la încărcarea imaginii: {e}")
        # Generează o imagine neagră ca fallback în caz de eroare
        return np.zeros((400, 400, 3), dtype=np.uint8)


# --- Funcție ajutătoare pentru a afișa imagini ---
def afiseaza_imagini(imagini, titluri, titlu_principal):
    """Afișează o listă de imagini folosind Matplotlib."""
    numar_imagini = len(imagini)
    plt.figure(figsize=(15, 5 * (numar_imagini // 3 + 1)))
    plt.suptitle(titlu_principal, fontsize=16)
    for i in range(numar_imagini):
        plt.subplot(1, numar_imagini, i + 1)
        plt.imshow(imagini[i], cmap='gray' if len(imagini[i].shape) == 2 else None)
        plt.title(titluri[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Sarcina 2: Deschide o imagine, afișează dimensiunea și o plotează ---
def sarcina_2(img):
    """Rezolvă sarcina 2 din laborator."""
    print("--- Sarcina 2 ---")
    if img is not None:
        print(f"Dimensiunea imaginii (înălțime, lățime, canale): {img.shape}")
        afiseaza_imagini([img], ["Imaginea Originală (lena.tif)"], "Sarcina 2: Deschidere Imagine")
        # Salvarea imaginii (opțional)
        # Convertim înapoi la BGR pentru salvare cu OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("lena_salvata.jpg", img_bgr)
        print("Imaginea a fost salvată ca 'lena_salvata.jpg'")
    else:
        print("Nu s-a putut încărca imaginea pentru Sarcina 2.")
    print("\n")


# --- Sarcina 3: Aplică filtre de blur și sharpen ---
def sarcina_3(img):
    """Rezolvă sarcina 3 din laborator."""
    print("--- Sarcina 3 ---")
    # 1. Blur (estompare) - folosim GaussianBlur
    # Testăm cu două valori diferite pentru kernel size (ksize)
    blur_1 = cv2.GaussianBlur(img, (5, 5), 0)
    blur_2 = cv2.GaussianBlur(img, (15, 15), 0)

    # 2. Sharpen (accentuare) - folosim un kernel custom
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpen_1 = cv2.filter2D(img, -1, kernel_sharpen)

    # Un kernel de sharpen mai agresiv
    kernel_sharpen_2 = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
    sharpen_2 = cv2.filter2D(img, -1, kernel_sharpen_2)

    afiseaza_imagini(
        [img, blur_1, blur_2, sharpen_1, sharpen_2],
        ["Original", "Blur (5x5)", "Blur (15x15)", "Sharpen (Tip 1)", "Sharpen (Tip 2)"],
        "Sarcina 3: Filtre de Blur și Sharpen"
    )
    print("Filtrele de blur și sharpen au fost aplicate și afișate.")
    print("\n")


# --- Sarcina 4: Aplică filtrul custom ---
def sarcina_4(img):
    """Rezolvă sarcina 4 din laborator."""
    print("--- Sarcina 4 ---")
    # Testăm diferite valori pentru a și b
    # Cazul 1: a=0, b=0 (similar cu un filtru de sharpen)
    w1 = np.array([[0, 0, 0],
                   [-2, 9, 0],
                   [0, -2, 0]], dtype=np.float32)
    filtru_1 = cv2.filter2D(img, -1, w1)

    # Cazul 2: a=-2, b=-2 (accentuează contururile verticale)
    w2 = np.array([[0, -2, 0],
                   [-2, 9, -2],
                   [0, -2, 0]], dtype=np.float32)
    filtru_2 = cv2.filter2D(img, -1, w2)

    # Cazul 3: a=2, b=2 (un efect de blur combinat cu sharpen)
    w3 = np.array([[0, 2, 0],
                   [-2, 9, 2],
                   [0, -2, 0]], dtype=np.float32)
    filtru_3 = cv2.filter2D(img, -1, w3)

    afiseaza_imagini(
        [img, filtru_1, filtru_2, filtru_3],
        ["Original", "Filtru (a=0, b=0)", "Filtru (a=-2, b=-2)", "Filtru (a=2, b=2)"],
        "Sarcina 4: Filtru Custom"
    )
    print("Efectele filtrului custom cu diferite valori a,b au fost afișate.")
    print("\n")


# --- Sarcina 5: Rotește o imagine ---
def sarcina_5(img):
    """Rezolvă sarcina 5 din laborator."""
    print("--- Sarcina 5 ---")
    (h, w) = img.shape[:2]
    centru = (w // 2, h // 2)

    # Rotație 45 de grade în sensul acelor de ceasornic (unghi negativ)
    M_45_cw = cv2.getRotationMatrix2D(centru, -45, 1.0)
    rotit_45_cw = cv2.warpAffine(img, M_45_cw, (w, h))

    # Rotație 90 de grade în sens invers acelor de ceasornic
    M_90_ccw = cv2.getRotationMatrix2D(centru, 90, 1.0)
    rotit_90_ccw = cv2.warpAffine(img, M_90_ccw, (w, h))

    afiseaza_imagini(
        [img, rotit_45_cw, rotit_90_ccw],
        ["Original", "Rotit 45° (CW)", "Rotit 90° (CCW)"],
        "Sarcina 5: Rotația Imaginii"
    )
    print("Cum se poate implementa o funcție de rotație?")
    print(
        "O funcție de rotație se bazează pe o transformare afină. Pentru fiecare pixel (x, y) din imaginea destinație,")
    print(
        "se calculează coordonatele corespunzătoare (x', y') în imaginea sursă folosind o matrice de rotație inversă.")
    print("Matricea de rotație 2D este de forma [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]].")
    print(
        "Deoarece pixelii calculați (x', y') pot cădea între pixeli reali, se folosește o tehnică de interpolare (ex: biliniară) pentru a determina culoarea.")
    print("\n")


# --- Sarcina 6: Funcție de decupare (crop) ---
def crop_image(img, x, y, latime, inaltime):
    """Decupează o regiune rectangulară dintr-o imagine."""
    # Verificăm dacă dimensiunile decupajului sunt valide
    if x >= 0 and y >= 0 and x + latime <= img.shape[1] and y + inaltime <= img.shape[0]:
        return img[y:y + inaltime, x:x + latime]
    else:
        print("Atenție: Coordonatele de decupare depășesc dimensiunile imaginii.")
        return None


def sarcina_6(img):
    """Rezolvă sarcina 6 din laborator."""
    print("--- Sarcina 6 ---")
    decupaj = crop_image(img, 100, 120, 250, 200)
    if decupaj is not None:
        afiseaza_imagini(
            [img, decupaj],
            ["Original", "Decupaj (100,120, w=250, h=200)"],
            "Sarcina 6: Decuparea Unei Imagini"
        )
        print("Funcția de decupare a fost aplicată.")
    print("\n")


# --- Sarcina 7: Creează un emoticon ---
def sarcina_7():
    """Rezolvă sarcina 7 din laborator."""
    print("--- Sarcina 7 ---")
    # Creăm o imagine goală (fundal alb)
    emoji = np.full((300, 300, 3), 255, dtype=np.uint8)

    # Fața (cerc galben)
    cv2.circle(emoji, (150, 150), 120, (0, 255, 255), -1)  # BGR: Galben = (0, 255, 255)
    cv2.circle(emoji, (150, 150), 120, (0, 0, 0), 3)

    # Ochiul stâng (cerc negru)
    cv2.circle(emoji, (110, 120), 20, (0, 0, 0), -1)

    # Ochiul drept (cerc negru)
    cv2.circle(emoji, (190, 120), 20, (0, 0, 0), -1)

    # Gura (zâmbet - arc de elipsă)
    cv2.ellipse(emoji, (150, 190), (60, 40), 0, 0, 180, (0, 0, 0), 5)

    # Convertim în RGB pentru afișare
    emoji_rgb = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)

    afiseaza_imagini([emoji_rgb], ["Emoji Creat"], "Sarcina 7: Creare Emoji")

    # Salvarea imaginii
    cv2.imwrite("numele_meu.jpg", emoji)
    print("Emoji-ul a fost creat și salvat ca 'numele_meu.jpg'.")
    print("\n")


# --- Execuția principală ---
if __name__ == '__main__':
    # URL pentru imaginea 'lena'. 
    # Exemplu: imagine = cv2.imread('cale/catre/lena.tif')
    URL_IMAGINE = "https://placehold.co/512x512/FF69B4/FFFFFF.png?text=Lene"
    imaginea_mea = incarcare_imagine_test(URL_IMAGINE)

    if imaginea_mea is not None:
        sarcina_2(imaginea_mea)
        sarcina_3(imaginea_mea)
        sarcina_4(imaginea_mea)
        sarcina_5(imaginea_mea)
        sarcina_6(imaginea_mea)
        sarcina_7()
    else:
        print("Scriptul nu a putut rula deoarece imaginea de test nu a fost încărcată.")

