# Laborator 1 - Viziune Computerizată - Toate Sarcinile
# Acest script conține soluțiile pentru sarcinile 2-7 din fișa de laborator.
# pip install opencv-python numpy matplotlib PyQt5

import cv2
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # Forțăm un backend grafic compatibil
import matplotlib.pyplot as plt
import os


# --- Funcții ajutătoare ---

def incarcare_imagine_test(nume_fisier="lena.tif"):
    """Încarcă imaginea de test. Dacă nu o găsește, returnează None."""
    if not os.path.exists(nume_fisier):
        print(f"EROARE: Fișierul imagine '{nume_fisier}' nu a fost găsit.")
        print("Vă rugăm să descărcați imaginea și să o plasați în folderul proiectului.")
        return None

    img = cv2.imread(nume_fisier)
    if img is None:
        print(f"EROARE: Nu s-a putut încărca imaginea '{nume_fisier}'.")
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def afiseaza_imagini(imagini, titluri, titlu_principal):
    """Afișează o listă de imagini folosind Matplotlib."""
    numar_imagini = len(imagini)
    plt.figure(figsize=(15, 5 * ((numar_imagini - 1) // 3 + 1)))
    plt.suptitle(titlu_principal, fontsize=16)
    for i in range(numar_imagini):
        plt.subplot((numar_imagini - 1) // 3 + 1, 3, i + 1)
        plt.imshow(imagini[i], cmap='gray' if len(imagini[i].shape) == 2 else None)
        plt.title(titluri[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Rezolvarea Sarcinilor ---

def sarcina_2(img):
    """Deschide o imagine, afișează dimensiunea și o plotează."""
    print("--- Sarcina 2: Deschidere și Afișare Imagine ---")
    if img is not None:
        inaltime, latime, canale = img.shape
        print(f"Dimensiunea imaginii: {latime}x{inaltime}, Canale: {canale}")
        afiseaza_imagini([img], ["Imaginea Originală"], "Sarcina 2")
    print("\n")


def sarcina_3(img):
    """Aplică filtre de blur și sharpen."""
    print("--- Sarcina 3: Filtre de Blur și Sharpen ---")
    if img is None: return

    # Blur (Filtru de medie)
    blurred_5x5 = cv2.blur(img, (5, 5))
    blurred_15x15 = cv2.blur(img, (15, 15))

    # Sharpen
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpen)

    afiseaza_imagini([img, blurred_5x5, blurred_15x15, sharpened],
                     ["Original", "Blur 5x5", "Blur 15x15", "Sharpen"],
                     "Sarcina 3")
    print("\n")


def sarcina_4(img):
    """Aplică un filtru custom."""
    print("--- Sarcina 4: Filtru Custom ---")
    if img is None: return

    # Testăm cu a=1, b=1
    kernel_custom1 = np.array([[0, 1, 0],
                               [-2, 9, 1],
                               [0, -2, 0]], dtype=np.float32)
    custom_img1 = cv2.filter2D(img, -1, kernel_custom1)

    # Testăm cu a=-2, b=-2 (efect de sharpen mai puternic)
    kernel_custom2 = np.array([[0, -2, 0],
                               [-2, 9, -2],
                               [0, -2, 0]], dtype=np.float32)
    custom_img2 = cv2.filter2D(img, -1, kernel_custom2)

    afiseaza_imagini([img, custom_img1, custom_img2],
                     ["Original", "Filtru (a=1, b=1)", "Filtru (a=-2, b=-2)"],
                     "Sarcina 4")
    print("\n")


def sarcina_5(img):
    """Rotește o imagine."""
    print("--- Sarcina 5: Rotație Imagine ---")
    if img is None: return

    inaltime, latime = img.shape[:2]
    centru = (latime / 2, inaltime / 2)

    matrice_rotatie_45 = cv2.getRotationMatrix2D(center=centru, angle=45, scale=1)
    img_rotita_45 = cv2.warpAffine(img, matrice_rotatie_45, (latime, inaltime))

    matrice_rotatie_neg_90 = cv2.getRotationMatrix2D(center=centru, angle=-90, scale=1)
    img_rotita_neg_90 = cv2.warpAffine(img, matrice_rotatie_neg_90, (latime, inaltime))

    afiseaza_imagini([img, img_rotita_45, img_rotita_neg_90],
                     ["Original", "Rotit 45°", "Rotit -90°"],
                     "Sarcina 5")
    print("\n")


def sarcina_6(img):
    """Decupează o parte dreptunghiulară din imagine."""
    print("--- Sarcina 6: Decupare (Crop) ---")
    if img is None: return

    # Parametri: y_start, y_end, x_start, x_end
    # Decupăm o regiune de 200x250 pixeli, începând de la (100, 150)
    x_start, y_start, w, h = 150, 100, 250, 200
    img_decupata = img[y_start: y_start + h, x_start: x_start + w]

    afiseaza_imagini([img, img_decupata],
                     ["Original", f"Decupat de la ({x_start},{y_start})"],
                     "Sarcina 6")
    print("\n")


def sarcina_7_creeaza_emoji(dimensiune=512):
    """Creează o imagine a unui emoji care plânge de râs."""
    print("--- Sarcina 7: Creare Emoji ---")
    canvas = np.ones((dimensiune, dimensiune, 3), dtype=np.uint8) * 255
    galben, negru, albastru, alb = (0, 255, 255), (0, 0, 0), (255, 100, 0), (255, 255, 255)
    centru = (dimensiune // 2, dimensiune // 2)
    raza_fata = dimensiune // 2 - 20

    cv2.circle(canvas, centru, raza_fata, galben, -1)
    cv2.circle(canvas, centru, raza_fata, negru, 4)

    offset_ochi_y, offset_ochi_x = -dimensiune // 12, dimensiune // 5
    dim_ochi = (dimensiune // 8, dimensiune // 10)
    centru_ochi_stang = (centru[0] - offset_ochi_x, centru[1] + offset_ochi_y)
    centru_ochi_drept = (centru[0] + offset_ochi_x, centru[1] + offset_ochi_y)
    cv2.ellipse(canvas, centru_ochi_stang, dim_ochi, 0, 0, 360, negru, -1)
    cv2.ellipse(canvas, centru_ochi_drept, dim_ochi, 0, 0, 360, negru, -1)

    dim_sprancene = (dimensiune // 6, dimensiune // 10)
    cv2.ellipse(canvas, centru_ochi_stang, dim_sprancene, 30, 180, 360, negru, 8)
    cv2.ellipse(canvas, centru_ochi_drept, dim_sprancene, -30, 180, 360, negru, 8)

    offset_gura_y = dimensiune // 6
    dim_gura = (dimensiune // 3, dimensiune // 5)
    centru_gura = (centru[0], centru[1] + offset_gura_y)
    cv2.ellipse(canvas, centru_gura, dim_gura, 0, 0, 180, negru, -1)
    cv2.ellipse(canvas, centru_gura, (dim_gura[0], dim_gura[1] - 25), 0, 0, 180, alb, -1)

    dim_lacrimi = (dimensiune // 10, dimensiune // 6)
    offset_lacrimi_y = dimensiune // 14
    centru_lacrima_stanga = (centru_ochi_stang[0], centru_ochi_stang[1] + offset_lacrimi_y)
    centru_lacrima_dreapta = (centru_ochi_drept[0], centru_ochi_drept[1] + offset_lacrimi_y)
    cv2.ellipse(canvas, centru_lacrima_stanga, dim_lacrimi, 0, 0, 360, albastru, -1)
    cv2.ellipse(canvas, centru_lacrima_stanga, dim_lacrimi, 0, 0, 360, negru, 3)
    cv2.ellipse(canvas, centru_lacrima_dreapta, dim_lacrimi, 0, 0, 360, albastru, -1)
    cv2.ellipse(canvas, centru_lacrima_dreapta, dim_lacrimi, 0, 0, 360, negru, 3)

    nume_fisier = "emoji_generat.jpg"
    cv2.imwrite(nume_fisier, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Emoji-ul a fost salvat ca '{nume_fisier}'")

    afiseaza_imagini([cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)], ["Emoji Generat"], "Sarcina 7")
    print("\n")


# --- Execuția Principală ---
if __name__ == '__main__':
    # **IMPORTANT**: Asigură-te că fișierul 'lena.tif' se află în același folder cu scriptul!
    imagine_test = incarcare_imagine_test()

    if imagine_test is not None:
        sarcina_2(imagine_test)
        sarcina_3(imagine_test)
        sarcina_4(imagine_test)
        sarcina_5(imagine_test)
        sarcina_6(imagine_test)

    # Sarcina 7 rulează independent de imaginea de test
    sarcina_7_creeaza_emoji()

