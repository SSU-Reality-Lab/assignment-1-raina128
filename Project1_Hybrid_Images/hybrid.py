import sys
import cv2
import numpy as np
import os

def gaussian_blur_kernel_2d(sigma, height, width):
    '''주어진 sigma와 (height x width) 차원에 해당하는 가우시안 블러 커널을
    반환합니다. width와 height는 서로 다를 수 있습니다.

    입력(Input):
        sigma:  가우시안 블러의 반경(정도)을 제어하는 파라미터.
                본 과제에서는 높이와 너비 방향으로 대칭인 원형 가우시안(등방성)을 가정합니다.
        width:  커널의 너비.
        height: 커널의 높이.

    출력(Output):
        (height x width) 크기의 커널을 반환합니다. 이 커널로 이미지를 컨볼브하면
        가우시안 블러가 적용된 결과가 나옵니다.
    '''
    # 커널의 중심 좌표
    cy = (height-1)/2.0
    cx = (width-1)/2.0
    
    # 좌표 그리드 만들기
    y = np.arange(height).reshape(-1,1)
    x = np.arange(width).reshape(1,-1)
    dy = y-cy # y좌표 중심으로부터의 거리
    dx = x-cx # x좌표 중심으로부터의 거리
    
    # 가우시안 공식 적용
    kernel = np.exp(-(dx*dx+dy*dy)/(2.0*sigma*sigma))
    
    # 합이 1이 되도록 정규화 (필터 적용 후 밝기 변화 없애기 위함)
    kernel /= kernel.sum()
    
    return kernel


def cross_correlation_2d(img, kernel):
    '''주어진 커널(크기 m x n )을 사용하여 입력 이미지와의
    2D 상관(cross-correlation)을 계산합니다. 출력은 입력 이미지와 동일한 크기를
    가져야 하며, 이미지 경계 밖의 픽셀은 0이라고 가정합니다. 입력이 RGB 이미지인
    경우, 각 채널에 대해 커널을 별도로 적용해야 합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).
    '''
    
    '''출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

    orig_dtype = img.dtype
    img_f = img.astype(np.float32, copy=False) # 이미지 float32 변환
    ker = kernel.astype(np.float32, copy=False) # 이미지 float32 변환

    # 커널 크기
    kh, kw = ker.shape
    ph = kh // 2
    pw = kw // 2

    if img_f.ndim == 2:  # grayscale
        H, W = img_f.shape
        padded = np.zeros((H + 2*ph, W + 2*pw), dtype=np.float32)
        padded[ph:ph+H, pw:pw+W] = img_f
        out = np.zeros((H, W), dtype=np.float32)

        for y in range(H):
            for x in range(W):
                acc = 0.0
                for i in range(kh):
                    for j in range(kw):
                        acc += padded[y + i, x + j] * ker[i, j]
                out[y, x] = acc

    else:  # RGB
        H, W, C = img_f.shape
        padded = np.zeros((H + 2*ph, W + 2*pw, C), dtype=np.float32)
        padded[ph:ph+H, pw:pw+W, :] = img_f
        out = np.zeros((H, W, C), dtype=np.float32)

        for c in range(C):
            for y in range(H):
                for x in range(W):
                    acc = 0.0
                    for i in range(kh):
                        for j in range(kw):
                            acc += padded[y + i, x + j, c] * ker[i, j]
                    out[y, x, c] = acc

    # 원래 데이터 타입으로 변환
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out, 0, 255).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype, copy=False)

    return out



def convolve_2d(img, kernel):
    '''cross_correlation_2d()를 사용하여 2D 컨볼루션을 수행합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    # 컨볼루션을 위해 커널을 상하좌우 뒤집은 뒤 cross-correlation 수행
    flipped = kernel[::-1, ::-1]
    return cross_correlation_2d(img, flipped)


def low_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 저역통과(low-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 저역통과 필터는 이미지의
    고주파(세밀한 디테일) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    
    if sigma <= 0:
        raise ValueError("sigma는 0보다 커야 합니다.")
    if size < 1 or size % 2 == 0:
        raise ValueError("size는 1 이상의 홀수여야 합니다.")

    # 등방성 가우시안 커널 (size x size)
    kernel = gaussian_blur_kernel_2d(sigma=sigma, height=size, width=size)

    # 컨볼루션으로 저역통과 필터 적용
    return convolve_2d(img, kernel)


def high_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 고역통과(high-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 고역통과 필터는 이미지의
    저주파(거친 형태) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    
    if sigma <= 0:
        raise ValueError("sigma는 0보다 커야 합니다.")
    if size < 1 or size % 2 == 0:
        raise ValueError("size는 1 이상의 홀수여야 합니다.")

    # 1) 가우시안 저역통과 생성 및 적용
    kernel = gaussian_blur_kernel_2d(sigma=sigma, height=size, width=size)
    low = convolve_2d(img, kernel)  # 금지 함수 없이 직접 구현한 컨볼루션 사용

    # 2) 원본 - 저역통과 = 고주파 성분
    orig_dtype = img.dtype
    out = img.astype(np.float32, copy=False) - low.astype(np.float32, copy=False)

    # 3) dtype 복원
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out, 0, 255).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype, copy=False)

    return out


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    # dtype 변환 (0~1 범위)
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0

    # 크기 맞추기
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 채널 수 맞추기
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 첫 번째 이미지 필터링
    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    # 두 번째 이미지 필터링
    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    # 두 이미지 혼합 (비율 기반)
    hybrid_img = (img1 * (1 - mixin_ratio) + img2 * mixin_ratio) * scale_factor
    
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
