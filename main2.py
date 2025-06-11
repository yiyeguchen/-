import cv2
# СЧИТЫВАНИЕ ИЗОБРАЖЕНИЯ:
img = cv2.imread('rock4.png')
cv2.imshow('Picture 1', img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('BW', gray)
cv2.waitKey(0)

# ФУНКЦИЯ НАХОЖДЕНИЯ КОНТУРОВ:
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
# ОТРИСОВКА РАЗЛИЧНЫХ КОНТУРОВ
for contour in contours:
    if i == 0:
        i = 1
        continue
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
    M = cv2.moments(contour)

for contour in contours:
    if i == 0:
        i = 1
        continue
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(gray, [contour], 0, (0, 0, 255), 5)
    M = cv2.moments(contour)

#ВЫВОД ИЗОБРАЖЕНИЯ:
cv2.imshow('shapes', img)

cv2.imshow('shapesbw', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#СОХРАНИТЬ ИЗОБРАЖЕНИЕ С КОНТУРАМИ:
cv2.imwrite('rock1111.png', img)

#ОБРАБОТКА ДВУХ БОЛЬШИХ КОНТУРОВ:
image = cv2.imread('rock1111.png')
original_image = image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50,200)
contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.destroyAllWindows()
def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas



#СОРТИРУЕМ СПИСОК В ОБРАТНОМ ПОРЯДКЕ (ПО УБЫВАНИЮ КОНТУРОВ)
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
#В ОТСОРТИРОВАННОМ ПО УБЫВАНИЮ СПИСКЕ ВЫБИРАЕМ ПЕРВЫЕ ЭЛЕМЕНТЫ - САМЫЕ БОЛЬШИЕ КОНТУРЫ:
largest_item= sorted_contours[0]
largest_item2= sorted_contours[1]

#ВЫВОД ИЗОБРАЖЕНИЯ:
cv2.drawContours(original_image, largest_item, -1, (255,0,0),3)
cv2.drawContours(original_image, largest_item2, -1, (255,0,0),3)
cv2.waitKey(0)
cv2.imshow('Largest Object', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()