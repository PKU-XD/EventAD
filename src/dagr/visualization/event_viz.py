import numba

# @numba.jit(nopython=True)
# def draw_events_on_image(img, x, y, p, alpha=0.5):
#     img_copy = img.copy()
#     for i in range(len(p)):
#         if y[i] < len(img):
#             img[y[i], x[i], :] = alpha * img_copy[y[i], x[i], :]
#             img[y[i], x[i], int(p[i])-1] += 255 * (1-alpha)
#     return img

@numba.jit(nopython=True)
def draw_events_on_image(img, x, y, p, alpha=0.5):
    img_copy = img.copy()
    for i in range(len(p)):
        if int(y[i]) < len(img):
            img[int(y[i]), int(x[i]), :] = alpha * img_copy[int(y[i]), int(x[i]), :]
            img[int(y[i]), int(x[i]), int(p[i])-1] += 255 * (1-alpha)
    return img