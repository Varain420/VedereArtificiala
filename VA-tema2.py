# Laborator 2 - Conversii de la RGB la Grayscale și Dithering


import cv2
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # Forțăm un backend grafic compatibil
import matplotlib.pyplot as plt
import os


# --- Funcție ajutătoare pentru a încărca o imagine LOCALĂ ---
def incarcare_imagine_locala(nume_fisier):
    """Încarcă o imagine de pe disc folosind OpenCV."""
    print(f"Încercare încărcare imagine locală: {nume_fisier}")
    if not os.path.exists(nume_fisier):
        print(f"EROARE: Fișierul imagine '{nume_fisier}' nu a fost găsit.")
        print(
            "Vă rugăm să descărcați o imagine colorată și să o salvați ca 'imagine_colorata.jpg' în folderul proiectului.")
        return None

    try:
        img = cv2.imread(nume_fisier)
        if img is None:
            raise Exception("Imaginea nu a putut fi încărcată. Verificați calea sau formatul fișierului.")
        # OpenCV încarcă imaginile în format BGR, le convertim în RGB pentru afișare corectă
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Imaginea locală a fost încărcată cu succes.")
        return img_rgb
    except Exception as e:
        print(f"Eroare la încărcarea imaginii locale: {e}")
        return None


# --- Funcție ajutătoare pentru a afișa imagini ---
def afiseaza_imagini(imagini, titluri, titlu_principal, cmap='gray'):
    """Afișează o listă de imagini folosind Matplotlib."""
    numar_imagini = len(imagini)
    plt.figure(figsize=(15, 8))
    plt.suptitle(titlu_principal, fontsize=16)
    for i in range(numar_imagini):
        plt.subplot((numar_imagini + 2) // 3, 3, i + 1)
        # Verificăm dacă imaginea este deja color sau trebuie afișată ca grayscale
        is_color = len(imagini[i].shape) == 3
        plt.imshow(imagini[i], cmap=None if is_color else cmap)
        plt.title(titluri[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Sarcina 1: Conversie prin Mediere Simplă ---
def sarcina_1_mediere_simpla(img_rgb):
    """
    Converteste imaginea la grayscale folosind media aritmetică a canalelor R, G, B.

    Explicație (R+G+B)/3 vs R/3+G/3+B/3:
    Când se lucrează cu tipuri de date întregi pe 8 biți (valori între 0-255), suma R+G+B
    poate depăși 255 (ex: 255+255+255=765), cauzând un 'overflow' dacă tipul de date
    intermediar nu este suficient de mare. Acest lucru duce la rezultate incorecte.
    Pentru a preveni asta, conversia la un tip de date mai mare (ex: float sau int16)
    este necesară înainte de adunare. Formula R/3 + G/3 + B/3, dacă este executată
    folosind operații pe floateri, evită această problemă de la bun început.
    """
    # Conversia la float pentru a evita overflow
    img_float = img_rgb.astype(np.float32)
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Calculăm media
    gray_img = (R + G + B) / 3.0

    # Convertim înapoi la uint8 pentru afișare
    return gray_img.astype(np.uint8)


# --- Sarcina 2: Conversie prin Medie Ponderată ---
def sarcina_2_medie_ponderata(img_rgb):
    """Convertește imaginea la grayscale folosind trei seturi de ponderi diferite."""
    weights1 = [0.299, 0.587, 0.114]  # Standardul (folosit și de OpenCV)
    weights2 = [0.2126, 0.7152, 0.0722]  # BT.709
    weights3 = [1 / 3, 1 / 3, 1 / 3]  # Echivalent cu medierea simplă, dar implementat cu dot product

    gray1 = np.dot(img_rgb[..., :3], weights1).astype(np.uint8)
    gray2 = np.dot(img_rgb[..., :3], weights2).astype(np.uint8)
    gray3 = np.dot(img_rgb[..., :3], weights3).astype(np.uint8)

    return gray1, gray2, gray3


# --- Sarcina 3: Conversie prin Desaturare ---
def sarcina_3_desaturare(img_rgb):
    """Gray = (min(R,G,B) + max(R,G,B)) / 2"""
    val_min = np.min(img_rgb, axis=2)
    val_max = np.max(img_rgb, axis=2)
    gray_img = ((val_min.astype(np.float32) + val_max.astype(np.float32)) / 2.0).astype(np.uint8)
    return gray_img


# --- Sarcina 4: Conversie prin Decompoziție ---
def sarcina_4_decompozitie(img_rgb):
    """Gray = max(R,G,B) și Gray = min(R,G,B)"""
    gray_max = np.max(img_rgb, axis=2)
    gray_min = np.min(img_rgb, axis=2)
    return gray_max, gray_min


# --- Sarcina 5: Conversie prin Canal Unic ---
def sarcina_5_canal_unic(img_rgb):
    """Extrage fiecare canal de culoare ca o imagine grayscale."""
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    return R, G, B


# --- Sarcina 6: Număr Custom de Nuanțe de Gri ---
def sarcina_6_nuante_gri_custom(img_gray, p=4):
    """Reduce numărul de nuanțe de gri la p."""
    factor_conversie = 255 / (p - 1)
    img_redusa = (np.round(img_gray / factor_conversie) * factor_conversie).astype(np.uint8)
    return img_redusa


# --- Sarcina 7: Dithering ---
def sarcina_7_dithering(img_gray, p=2, method='floyd'):
    """Aplică dithering Floyd-Steinberg sau Burkes pentru a reduce numărul de culori."""
    img_dither = img_gray.astype(np.float32)
    h, w = img_dither.shape

    if method == 'floyd':
        # * 7/16
        # 3/16 5/16 1/16
        mask = [(0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)]
    elif method == 'burkes':
        # * * 8/32 4/32
        # 2/32 4/32 8/32 4/32 2/32
        mask = [(0, 1, 8 / 32), (0, 2, 4 / 32),
                (1, -2, 2 / 32), (1, -1, 4 / 32), (1, 0, 8 / 32), (1, 1, 4 / 32), (1, 2, 2 / 32)]
    else:
        raise ValueError("Metoda de dithering necunoscută. Alegeți 'floyd' sau 'burkes'.")

    factor_conversie = 255 / (p - 1)

    for y in range(h):
        for x in range(w):
            old_pixel = img_dither[y, x]
            new_pixel = np.round(old_pixel / factor_conversie) * factor_conversie
            img_dither[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            for dy, dx, factor in mask:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img_dither[ny, nx] += quant_error * factor

    # Ne asigurăm că valorile rămân în intervalul [0, 255]
    img_dither = np.clip(img_dither, 0, 255)
    return img_dither.astype(np.uint8)


# --- Problema Inversă: Grayscale to RGB ---
def sarcina_8_grayscale_to_rgb(img_gray):
    """Transformă o imagine grayscale într-una colorată folosind o hartă de culori."""
    # O metodă simplă este să copiem canalul gri în R, G și B.
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # O metodă mai interesantă ("false color") folosește o hartă de culori (colormap)
    img_false_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

    return img_gray_3ch, img_false_color


# --- Execuția principală ---
if __name__ == '__main__':
   

    imagine_color = incarcare_imagine_locala('imagine_colorata.jpg')

    if imagine_color is not None:
        # --- Rulează sarcinile 1-5 ---
        gray1 = sarcina_1_mediere_simpla(imagine_color)
        gray2_1, gray2_2, gray2_3 = sarcina_2_medie_ponderata(imagine_color)
        gray3 = sarcina_3_desaturare(imagine_color)
        gray4_max, gray4_min = sarcina_4_decompozitie(imagine_color)
        gray5_R, gray5_G, gray5_B = sarcina_5_canal_unic(imagine_color)

        afiseaza_imagini([imagine_color, gray1, gray2_1, gray3, gray4_max, gray4_min, gray5_R, gray5_G, gray5_B],
                         ["Original", "1. Medie Simplă", "2. Medie Ponderată", "3. Desaturare",
                          "4. Max Decomp", "4. Min Decomp", "5. Canal Roșu", "5. Canal Verde", "5. Canal Albastru"],
                         "Metode de Conversie Grayscale")

        # --- Rulează sarcina 6 ---
        img_gray_standard = cv2.cvtColor(imagine_color, cv2.COLOR_RGB2GRAY)
        nuante_4 = sarcina_6_nuante_gri_custom(img_gray_standard, p=4)
        nuante_8 = sarcina_6_nuante_gri_custom(img_gray_standard, p=8)
        afiseaza_imagini([img_gray_standard, nuante_4, nuante_8],
                         ["Grayscale Original (256 nuanțe)", "Custom (4 nuanțe)", "Custom (8 nuanțe)"],
                         "Sarcina 6: Număr Custom de Nuanțe de Gri")

        # --- Rulează sarcina 7 ---
        dither_floyd = sarcina_7_dithering(img_gray_standard, p=2, method='floyd')
        dither_burkes = sarcina_7_dithering(img_gray_standard, p=2, method='burkes')
        binarizat_simplu = sarcina_6_nuante_gri_custom(img_gray_standard, p=2)  # Binarizare simplă (fără dithering)

        afiseaza_imagini([img_gray_standard, binarizat_simplu, dither_floyd, dither_burkes],
                         ["Grayscale Original", "Binarizat Simplu (2 culori)",
                          "Floyd-Steinberg Dithering", "Burkes Dithering"],
                         "Sarcina 7: Dithering (Conversie la imagine binară)")

        # --- Rulează sarcina 8 ---
        gray_3ch, false_color = sarcina_8_grayscale_to_rgb(img_gray_standard)
        afiseaza_imagini([img_gray_standard, gray_3ch, false_color],
                         ["Original Grayscale", "Copiat pe 3 Canale", "False Color (COLORMAP_JET)"],
                         "Problema Inversă: Grayscale to RGB", cmap='gray')
    else:
        print("Scriptul nu a putut rula deoarece imaginea de test nu a fost încărcată.")


