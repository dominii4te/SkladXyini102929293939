import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import onnxruntime as ort

# ─────────────────────────────────────────────
# Явно задаём провайдеры: CUDA в приоритете, CPU как fallback.
# Без этого insightface может тихо выбрать CPU даже если
# onnxruntime-gpu установлен и GPU доступен.
# ─────────────────────────────────────────────
def get_providers():
    available = ort.get_available_providers()
    print(f"[FaceSwapper] Available ONNX providers: {available}")
    if 'CUDAExecutionProvider' in available:
        print("[FaceSwapper] ✅ Using CUDA (GPU)")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("[FaceSwapper] ⚠️ CUDAExecutionProvider NOT available — running on CPU. "
              "Install onnxruntime-gpu: pip uninstall onnxruntime -y && pip install onnxruntime-gpu")
        return ['CPUExecutionProvider']

PROVIDERS = get_providers()


class FaceSwapper:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
        # ctx_id=0 selects GPU 0 when a CUDA provider is present; ignored for CPU-only.
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', download=True, download_zip=True, providers=PROVIDERS
        )
        # Кеш source-лица: для видео один и тот же source-путь запрашивается
        # сотни раз подряд (один кадр = один вызов). Без кеша детекция source
        # лица (тяжёлая операция) повторяется на каждом кадре впустую.
        self._source_cache_key = None
        self._source_cache_faces = None

    def _get_source_face(self, source_path, source_face_idx):
        cache_key = (source_path, os.path.getmtime(source_path))
        if cache_key != self._source_cache_key:
            source_img = cv2.imread(source_path)
            if source_img is None:
                raise ValueError("Could not read source image")
            faces = sorted(self.app.get(source_img), key=lambda x: x.bbox[0])
            self._source_cache_key = cache_key
            self._source_cache_faces = faces
        faces = self._source_cache_faces
        if len(faces) < source_face_idx or source_face_idx < 1:
            raise ValueError(f"Source image contains {len(faces)} faces, but requested face {source_face_idx}")
        return faces[source_face_idx - 1]

    def swap_faces(self, source_path, source_face_idx, target_path, target_face_idx):
        target_img = cv2.imread(target_path)
        if target_img is None:
            raise ValueError("Could not read target image")

        source_face = self._get_source_face(source_path, source_face_idx)

        target_faces = sorted(self.app.get(target_img), key=lambda x: x.bbox[0])
        if len(target_faces) < target_face_idx or target_face_idx < 1:
            raise ValueError(f"Target image contains {len(target_faces)} faces, but requested face {target_face_idx}")
        target_face = target_faces[target_face_idx - 1]

        result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
        return result

    def count_faces(self, img_path):
        img = cv2.imread(img_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces)


def main():
    source_path = os.path.join("SinglePhoto", "data_src.jpg")
    target_path = os.path.join("SinglePhoto", "data_dst.jpg")
    output_dir = os.path.join("SinglePhoto", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    swapper = FaceSwapper()

    try:
        try:
            user_input = input("Enter the target face index (starting from 1, default is 1): ")
            target_face_idx = int(user_input) if user_input.strip() else 1
            if target_face_idx < 1:
                print("Invalid index. Using default value 1.")
                target_face_idx = 1
        except ValueError:
            print("Invalid input. Using default value 1.")
            target_face_idx = 1

        try:
            result = swapper.swap_faces(
                source_path=source_path,
                source_face_idx=1,
                target_path=target_path,
                target_face_idx=target_face_idx
            )
        except ValueError as ve:
            if "Target image contains" in str(ve):
                print(f"Target face idx {target_face_idx} not found, trying with idx 1.")
                result = swapper.swap_faces(
                    source_path=source_path,
                    source_face_idx=1,
                    target_path=target_path,
                    target_face_idx=1
                )
            else:
                raise ve
        output_path = os.path.join(output_dir, "swapped_face.jpg")
        cv2.imwrite(output_path, result)
        print(f"Face swap completed successfully. Result saved to: {output_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
