const char layer1_w[10][784] = {
{-6, -3, 2, -8, 4, 0, -2, 2, -4, 2, -2, 3, -3, -1, -6, 3, 2, -3, 4, -5, -3, -7, 9, -2, 9, 9, 1, -7, -2, 0, 1, 0, 3, 0, -1, 5, 10, 9, 19, 19, 14, -4, -1, 9, 19, 14, 11, 6, 0, 6, 4, 3, -1, -7, -4, -3, 5, -8, 7, 7, -4, 11, 32, 25, -16, 13, 33, 28, 4, -13, 0, 6, 12, -16, -12, -41, -23, -19, -17, 11, 11, 8, 1, 0, -2, -4, 19, 2, -2, 17, 20, 33, -6, 2, -3, -29, -21, -13, -36, -24, -27, -66, -32, -25, -7, 4, 18, 14, 25, 17, -2, -3, 4, 4, 10, -1, -6, 9, 3, -13, 7, 0, -10, -20, -5, -19, 0, -8, -1, -6, -15, 0, 1, 8, -7, 5, 2, 18, 16, 1, -1, -2, 22, -2, -1, 11, 7, 4, -5, -16, -7, -9, -10, 4, -2, -23, -11, -12, -11, -11, -14, -8, -3, -3, 11, 21, 22, -6, 9, -4, 3, 26, 20, 8, 1, -7, 5, 14, 5, 8, 6, 5, 1, 19, 19, 10, -1, 3, 5, 0, -22, -10, -2, 16, 14, 13, 1, 10, 20, 25, 26, -14, -2, 19, 10, 4, 1, 4, 12, 4, 12, 7, 22, 5, 19, 5, 1, 18, -5, -9, -6, -26, -8, 10, -15, 30, 11, 28, 29, 7, 24, 20, 28, 11, 20, 9, 11, 22, 25, 16, 3, 11, 2, 3, 10, -1, 20, 4, -11, -12, 31, 37, 9, 28, 5, 17, 29, 22, 27, 19, 24, 21, 34, 32, 25, 31, 27, -1, -4, -4, -22, -10, -1, -5, 6, 14, 19, 11, 51, -3, 6, 18, 0, 11, 31, 36, 49, 25, 20, 30, 45, 45, 41, 52, 35, 2, -4, -17, 2, -6, -2, 10, 15, 26, 8, 30, 39, -1, -1, 8, 23, 15, 39, 46, 39, 27, 17, 22, 20, 36, 15, 41, 34, -1, -12, -9, 4, -2, -7, 3, 15, 20, 35, 42, 37, 18, 7, 20, 30, 7, 35, 23, 6, 5, 10, 8, -8, -30, -35, -40, 1, -9, -13, 6, 11, -1, 1, -13, -4, 4, -9, 11, 18, 11, -1, 16, 34, 12, 0, -8, -22, -17, -25, -23, -40, -62, -73, -46, -10, -15, 8, -1, -2, -9, 0, -8, -8, -27, -31, 5, 11, 24, 19, -6, 27, 9, -46, -64, -34, -60, -43, -53, -53, -64, -59, -3, 13, 1, 5, -8, -14, 3, 1, -9, -7, -25, -30, 9, 41, 23, 19, -6, -11, -21, -47, -59, -42, -52, -51, -57, -43, -35, -11, 28, 18, 7, -11, -3, 2, -7, -11, -1, -28, -33, -13, -17, 19, 26, 16, -22, -11, -21, -59, -58, -45, -48, -35, -29, -14, -11, 7, 26, 11, -1, 14, 0, -2, -2, -12, -12, -17, -15, -18, 4, 33, 28, 2, -4, -11, -18, -32, -41, -26, -17, -19, -9, -10, 8, 27, 18, 20, 6, 8, 5, 1, -6, -1, -12, -14, -8, -8, 14, 22, 39, 6, -1, -2, -15, -20, -13, 1, -12, -8, -5, -7, 4, 20, 12, 10, 13, -6, 11, 3, 0, -18, -2, -7, -20, 12, 15, 26, 27, -3, 13, -17, -23, -10, -13, -8, 5, 7, 3, 4, 1, 22, 13, 19, 11, 11, 9, 9, 7, -9, -17, -6, 6, 0, 30, 32, 25, 6, 14, 2, -1, -5, -17, 10, 7, 1, 0, 4, 3, 17, 16, 12, 0, 5, 0, 8, 3, 4, -4, -2, 4, 11, 25, 25, -5, 3, 15, -10, -5, -27, 2, 13, 8, -12, 2, -1, 1, -6, 14, 7, 16, -1, 5, 15, 8, 3, -2, 1, 5, 17, 24, -19, 7, 7, -3, -3, -3, 4, 23, 13, 0, 6, -6, -5, -16, 5, -9, 11, 13, -5, 13, 13, -2, 11, 9, 22, 20, 31, 18, -26, 3, -6, -4, -17, -41, 3, 10, 20, 3, 9, -11, -15, -5, 4, -4, -5, 11, 15, 8, 6, 16, 24, 12, 28, 30, 27, 2, -5, 2, -6, -10, 4, -37, -28, -28, -19, -11, -3, -5, -16, -2, 5, -1, 4, 28, 14, 15, 31, 29, 22, 6, -2, 7, 37, -5, 19, 7, -8, -5, -17, -12, -11, -38, -17, -24, 5, 4, -5, 14, 30, 21, 19, 35, 23, 36, 18, 11, 9, 2, -10, -6, 23, -5, 1, 8, -5, 4, 6, -18, -26, -27, -1, 16, 26, 27, 25, 33, 20, 27, 37, 23, 10, 29, 31, 21, 3, 5, 12, 0, -6, 4, -8, -1, -6, 7, 7, -4, 24, 18, 19, 15, 25, 21, 30, 11, 11, 38, 21, 5, 12, 23, 12, 39, 11, 8, 9, 19, -9, -4, -8, 4},
{4, -1, -9, -6, -5, 1, -2, 0, 1, 4, -7, 4, 9, 13, -10, -8, -7, -1, -6, 1, -8, -1, 10, 7, -9, -5, -5, 4, -5, 4, -9, 4, 1, 18, 17, 24, 18, 26, 36, 34, 53, 48, 15, 35, 12, 28, 37, 21, 19, 17, 15, 20, -5, 0, -5, -1, 9, -3, 8, 12, 17, 21, 20, 37, 31, 10, 45, 27, 39, 7, 5, 0, 14, 30, 39, 41, 31, 14, 17, 31, -17, -1, 9, 1, 3, 5, -1, 22, 24, 10, 14, 22, 30, 28, 31, 19, 29, 34, 27, 33, 25, 27, 28, 35, 54, 29, 5, 13, 10, 6, 11, 6, 5, 8, -19, 35, -11, -8, 2, 8, -2, 6, 10, 28, 7, -3, 12, 21, 15, 17, 24, 5, -11, 3, 14, 13, -1, 1, 31, 9, 6, -7, -16, 22, -24, -5, 4, 10, 3, 17, 12, -21, -12, -8, -15, -6, 2, 15, 7, -4, 7, 2, 1, -2, 10, 32, 29, 23, 9, 8, 25, 13, 8, 24, 18, 17, 24, 6, 7, -1, 2, -11, -10, -6, 1, -10, 5, -1, -1, 2, 7, 2, 32, 26, 5, 7, -5, 26, 8, 22, 1, 24, 29, 22, 12, 9, 16, 10, 3, -7, -9, -14, -7, 0, 8, 1, 2, 6, 10, 22, 42, 35, -10, 25, 8, 30, 22, 29, -2, 6, 2, 24, 16, 17, 14, 22, 18, 3, 11, 4, 8, 14, 4, 4, 21, -2, 26, 27, 32, 28, 3, -11, 18, 29, 14, 15, 25, 10, 7, 6, 10, 8, 2, 12, 14, 20, 14, -4, -2, 9, 6, 6, 7, 25, 17, 27, 65, 51, 9, 21, 12, 17, 30, 39, 18, 5, -1, 14, 10, 11, 2, 10, 8, 27, 37, 6, 5, 2, 1, -2, 9, 5, 10, 26, 56, 65, 55, 11, 21, 26, 54, 27, 22, -11, -5, 12, -3, 10, 5, 6, 11, 39, 12, -5, -17, -7, -10, -8, -19, -17, -28, -16, 14, 47, 39, 0, 7, 23, 41, 39, 12, -8, 18, 14, 12, 3, 12, 16, 11, 7, -9, -23, -18, -10, -7, -28, -24, -43, -36, -41, -29, 5, -4, -23, -3, 2, 32, 28, -11, 13, 12, 9, 6, 12, 18, 7, -7, -10, -23, -32, -29, -11, -12, -17, -17, -19, -5, 1, 19, -23, -52, -19, 9, 1, 23, 6, -21, 4, -5, 0, 8, 12, 12, -2, -14, -28, -30, -16, -28, -11, 4, -6, 2, 5, 19, 40, 45, -12, -51, -8, 12, 18, 6, -13, -15, -24, -5, 2, 0, 11, 5, -9, -9, -20, -27, -10, -26, -3, 16, 10, 31, 26, 28, 44, 47, -15, -49, -34, 10, 9, 10, 0, -7, -13, -9, -12, -16, 1, -15, -8, -16, -18, -12, -18, -16, -8, 13, 31, 22, 12, 13, 26, 34, -37, -39, -24, 2, 20, -8, 21, 1, -3, -19, -24, -33, -25, -20, -24, -27, -37, -26, -22, -3, 11, 11, 22, 22, 17, 20, 15, 10, -19, -19, 1, 18, -5, 13, 35, 10, 8, -2, -29, -19, -30, -41, -26, -32, -15, -5, 3, 19, 17, 8, 12, 12, 20, 30, 6, -10, -26, -11, -31, 0, 17, 10, 39, 22, 1, 1, -1, 11, 1, 1, 0, 19, 18, 9, 17, 1, 12, 15, 0, 13, 16, 18, 9, 4, -40, -41, -20, 6, 26, -4, 22, 4, 20, -1, -1, 16, 21, 22, 24, 33, 21, 21, 17, 6, 12, 5, -7, -5, -6, 15, 6, -16, -28, -22, 9, -9, -1, -14, 14, 5, -3, 8, 12, 14, -3, 12, 19, 31, 31, 25, 18, 9, 10, -7, -5, -5, -10, 0, -14, -32, -27, 5, 11, 5, -5, 2, 0, 2, -9, -7, -3, 1, 7, -4, 10, 11, 17, 19, 27, 22, 3, 11, 1, -17, -11, -9, -22, -32, -33, 19, 15, 6, 7, 25, 18, 29, 29, 21, 16, 3, 19, 17, 20, 12, 11, 12, 19, 9, 10, -8, -21, -16, -41, -34, -28, -14, -18, 9, 0, -9, 4, 19, 61, 56, 34, 32, 5, 17, 23, 10, 17, 16, 32, 33, 21, 2, 2, -2, -25, -34, -35, -21, -31, -14, -36, -16, 7, 8, 2, -20, 37, 43, 7, 23, 27, 25, 23, 19, 25, 26, 31, 7, 18, 27, 5, 14, 3, -25, -9, 9, 15, 18, -3, 2, -2, 9, 5, -6, 5, -18, -10, -17, -9, -6, -16, -11, 0, -1, 3, -22, 10, 13, -6, 7, 35, 4, 6, 19, 2, -13, 6, 4, 9, -3, 6, -1, 3, 25, 35, 4, -1, 0, 10, 12, 7, -3, 41, 7, 7, 7, 45, 21, 16, 2, 18, 16, 29, 4, 5, -8, -1},
{10, 3, 9, 10, -2, 6, 1, 4, -8, 1, 3, 0, 10, 10, -1, -6, -3, -2, 5, -4, -4, -9, 5, 4, -2, -1, -3, 4, 2, 0, 8, -1, -1, 12, 11, 24, 26, 16, 17, 32, 15, 21, -20, 8, 20, 32, 41, 18, 33, 17, 21, 12, 7, -7, 1, 3, 4, -6, 10, 4, 12, 6, 5, 20, 22, 16, 15, 1, 14, 9, -29, -32, -16, 12, 14, 17, 1, 1, 21, 12, -16, -2, -6, -9, -5, 2, -9, 15, 7, -23, -34, -30, 12, 0, 0, 14, 12, 27, 29, 18, 1, 13, 22, 8, 16, 7, 3, 7, -4, -9, 33, -3, 4, -12, -33, -7, -51, -62, -30, -15, -8, 13, 18, 31, 31, 13, 25, 24, 11, 16, 20, -14, -22, -6, -8, -9, 4, 16, 28, 26, 5, -3, -28, -28, -73, -81, -41, -22, -10, -7, 13, 38, 36, 47, 35, 24, 19, 12, 16, 8, 12, -2, -18, -28, -11, 26, 47, 26, 10, -16, 8, -50, -59, -61, -50, -29, -22, -8, 4, 21, 33, 45, 41, 51, 33, 42, 32, 21, 8, -9, -8, -14, 6, 34, 2, 3, -1, -32, -21, -38, -72, -49, -29, -34, -7, 5, 4, 16, 27, 36, 50, 38, 41, 22, 8, 12, 7, 6, -14, 3, 21, 37, -16, 26, 13, -13, -14, -40, -58, -38, -23, -24, 8, 8, 12, 10, 18, 14, 31, 25, 24, 12, 3, 3, 4, 11, 1, 26, 49, 43, 11, 9, -9, -6, -18, -36, -54, -25, 4, 0, 11, 17, 19, 13, 15, 14, 8, -2, -21, -18, -8, -7, -2, 1, -2, 39, 78, 54, 22, 30, -14, -22, -33, -34, -29, -10, 20, 22, 31, 10, 22, 24, 9, 2, 17, -18, -32, -17, -13, -11, -1, -7, 2, 40, 76, 80, 62, 16, -23, -10, -30, -37, -4, 9, 37, 22, 24, 9, -5, -11, -10, 37, 19, -24, -26, -14, -16, -8, -4, -5, -14, 37, 62, 78, 54, 8, -6, -25, -27, -12, 5, 10, 26, 5, -9, 1, -11, -18, -2, 35, 33, -5, -39, -12, -3, 3, 8, -7, -6, 10, -5, -21, 7, 16, -2, -10, -33, -5, 19, 5, -4, -2, -5, 1, -12, -6, 21, 50, 23, 3, -21, -11, -8, 12, -8, 1, 5, 11, -6, -43, -4, -2, 17, -13, -30, -15, -6, -11, -18, -5, -4, 1, 6, 2, 35, 42, 20, -1, -13, -27, -16, -32, -5, -8, -14, -11, 0, -39, -28, 4, 3, 14, -39, -28, -16, -20, -23, -13, -11, -4, 16, 22, 25, 17, -4, -17, -13, -20, -10, -18, -6, -10, -37, -8, 8, -14, -62, -35, -7, 3, -43, -10, -5, -17, -26, -24, -11, -8, 27, 31, 22, 0, -13, -28, -22, -20, -13, -22, -23, -34, -10, -15, -14, -19, -44, -36, -1, -16, -48, -6, -17, 5, 8, -14, -15, 13, 24, 28, 3, -20, -38, -16, -16, -11, -4, -8, -5, -3, -19, -1, -9, -20, -28, 0, 7, -16, -42, -34, -14, -6, 26, 9, 7, 14, 39, 25, -8, -24, -17, -10, 0, 2, -1, 7, 7, 4, 4, 11, 10, -2, -36, -25, -6, -22, -5, -28, -1, 1, 37, 29, 41, 31, 33, 29, 18, 8, -9, -13, 8, 1, 13, 27, 21, 19, 34, 31, 31, 6, -42, -20, 1, 10, -5, -10, 3, 32, 19, 18, 31, 31, 42, 25, 11, 23, 0, 5, 10, 20, 23, 30, 17, 26, 44, 21, 31, 36, -35, 3, -11, -27, -8, 16, 29, 32, 34, 39, 20, 28, 15, 13, 20, 17, 7, 12, 9, 2, 6, 10, 23, 31, 41, 29, 37, 19, 20, 4, -5, -1, 0, 18, 14, 2, 9, 5, 19, 20, 5, 18, 25, 33, 22, 15, 9, 12, 1, -10, 20, 9, 34, 60, 32, 14, 24, 13, 4, 1, 18, 27, 6, -2, -28, -24, 12, 7, 18, 19, 24, 29, 18, 5, 5, -2, -2, 6, 17, 21, 50, 63, 20, 14, 1, 6, -2, -1, 1, 13, -9, -22, -15, -16, 10, 8, 15, 14, 14, 19, 7, 17, 11, 12, 21, 6, 30, 17, 26, 51, 31, -24, -19, 1, 8, 9, 28, -7, -6, 17, -2, 3, -3, -2, 3, 7, 6, 10, 10, 15, 24, 30, 22, 37, 40, 33, 23, 33, 35, -10, -5, 7, 1, 5, 0, 10, 47, 36, 29, -1, 7, 10, 17, 28, 44, 36, 34, 37, 36, 24, 11, 17, 28, 7, -16, -9, 0, -5, 7, -3, -5, 1, 6, -3, -19, -8, -27, -18, -30, 1, 6, 18, 18, -20, 7, 1, -14, -45, -24, -9, -6, -22, -29, -16, -2, 0, 9, 4},
{-4, -5, 9, 2, 5, -7, 7, -8, -5, -3, -6, -5, 15, 13, 16, -1, 7, 2, 5, -4, 5, 6, 4, -9, 0, -2, 10, 4, 4, 8, 7, 6, 14, 21, 31, 40, 27, 43, 38, 38, 36, 20, 10, 33, 64, 54, 68, 30, 28, 19, 18, 20, 5, -3, 4, 9, 6, -1, 4, -3, 10, 25, 42, 46, 62, 72, 64, 72, 57, 47, 56, 48, 59, 52, 37, 36, 27, 45, 33, 40, 22, 8, 6, 4, -4, 4, 4, 8, 11, -13, 19, 18, 25, 37, 44, 26, 26, 37, 12, 20, 21, 18, 8, 22, 0, 15, 33, 23, 10, 2, 9, -2, -7, -14, -23, -13, -12, -11, 8, -2, 7, 6, 15, 3, 11, -2, 7, 2, -1, 18, 6, 4, 13, 10, 2, 19, -31, -1, -13, 14, 4, 7, 22, -26, -13, -32, -2, -10, 9, 17, 8, 3, 11, 3, 12, 7, 14, 9, 21, 4, 11, 12, 13, 44, 19, -6, -3, 14, 7, -14, -7, -9, 13, 0, -8, 2, -3, 0, 1, -8, -7, -14, -7, -7, -13, -4, -19, -15, -12, -28, -13, -8, 13, 13, 12, 7, 5, -18, 0, -25, -6, -11, -14, 2, 4, -3, -4, 2, -14, -14, -20, -20, -26, -30, -24, -16, -23, -19, -6, -31, 7, 22, 15, 2, 26, 9, 7, -26, -3, -21, -25, -7, -12, -1, 5, -1, -11, -4, -15, -14, -21, -17, -31, -20, -21, -16, -21, -16, -17, -13, -14, -27, 9, -1, 0, -4, -24, -17, -17, -14, -2, -7, 3, 10, 2, -5, -24, -9, -22, -22, -27, -22, -24, -13, -23, -12, -22, -29, -23, 5, 9, 5, -13, -16, -17, -2, -3, -3, 8, 6, 6, 2, -7, -12, -15, -19, -6, -14, -30, -26, -23, -25, -28, -13, -5, -31, -17, -5, -3, 20, -23, -14, -13, 2, -1, -3, -10, 1, -16, -19, -16, -30, -38, -30, -5, -33, -31, -7, -29, -11, -17, -4, -18, -20, -41, -2, 3, -7, 5, -30, -21, -11, -20, -23, -2, -18, -21, 0, -11, -16, -29, -17, -12, -20, -15, -25, -22, 3, -11, -10, 20, 10, 1, 19, 0, -5, -11, -25, -55, -9, -14, -19, -8, -7, 16, 15, 16, 9, 3, -2, -3, -5, -18, -8, 2, 10, 10, 22, 35, 42, 48, 22, 8, -6, -7, -11, -40, -12, -6, 0, 15, 18, 15, 35, 15, 17, 8, 16, 15, -4, 2, 3, 11, 8, 28, 29, 33, 49, 68, 31, 16, -14, 1, 15, -11, 16, 14, 16, 11, 24, 29, 34, 25, 23, 30, 17, 16, 1, 8, 2, 4, 5, -10, -5, 30, 71, 41, 28, -9, -22, 6, 16, -1, 22, 17, 20, 9, 21, 36, 33, 14, 22, 22, 36, 23, 8, 9, 9, 3, 23, 3, -2, 33, 114, 65, 27, -7, -28, 15, 16, 3, 7, 26, 10, -1, 19, 27, 25, 9, 29, 32, 30, 14, -2, 10, 8, 6, 13, -1, 8, 24, 109, 61, 50, 7, -19, -6, -6, 6, 1, -5, 10, 10, -2, 14, 15, 36, 39, 34, 19, 18, 22, 14, 15, 14, 5, 4, 22, 54, 98, 28, 32, -9, -15, 0, -9, 8, -5, -4, 18, 6, 6, 17, 26, 35, 35, 29, 28, 26, 14, 15, 4, 5, 13, 15, 17, 23, 51, 38, 29, -6, -11, 22, 4, 10, -6, -5, 3, 2, 12, 29, 35, 35, 19, 27, 19, 19, 22, 22, 23, 11, 15, 28, 21, 23, 13, 20, -19, 9, 14, -4, 9, 2, 4, -7, -2, 7, 22, 28, 27, 26, 24, 28, 19, 16, 29, 30, 15, 15, 26, 22, 31, 31, 25, 26, -10, 8, -5, 6, 8, 18, 6, -9, -5, 12, 10, 16, -2, 19, 4, -1, 4, 10, 17, 15, 22, 27, 19, 17, -8, 8, 0, 1, 2, 3, -8, -13, -11, -6, 3, -15, -8, -3, 3, -11, -9, 5, -15, -15, -5, 5, -9, 4, 5, 23, 15, 4, -1, 19, 16, 0, -8, 2, -4, 11, -29, -18, -3, -24, -34, -38, -10, -29, -31, -25, -32, -21, -23, -25, -32, -18, -27, -14, -12, -19, -21, 42, 30, 16, -1, -8, -4, 4, -12, -17, -39, -35, -54, -86, -99, -83, -63, -61, -83, -69, -74, -81, -84, -89, -83, -68, -44, -33, -13, 5, 18, 21, 6, 2, 6, 5, 4, 10, 13, -7, -15, -22, -21, -21, -35, -42, -53, -52, -46, -67, -40, -77, -101, -81, -48, -25, -20, -3, 7, 2, -6, -8, -3, -9, -7, -2, 9, -3, -5, 8, 13, 8, -8, -20, -16, -33, -20, -5, -20, -5, -21, 0, 5, 0, -6, 1, -6, -1, -2},
{-5, -5, -1, 2, 8, -2, 1, -6, 5, 9, -9, 6, -16, -9, 3, -6, -1, 4, -8, -5, -7, 5, 1, -1, 3, 4, 8, 7, 5, -3, 0, 1, 3, -10, -30, -29, -33, -39, -41, -41, -46, -26, 5, -26, -10, -25, -42, -32, -19, -25, -14, -12, -9, -4, -2, 9, -4, -1, -3, -21, -33, -19, -29, -42, -52, -44, -50, -58, -61, -50, -68, -56, -34, -8, -19, -34, -32, -42, -37, -39, 0, 18, 0, -4, -7, -4, -11, -31, -10, -11, -43, -46, -41, -46, -43, -41, -40, -41, -48, -37, -34, -17, -14, -8, -10, 1, -3, -23, -11, 16, -1, -2, -9, -1, -24, -19, -36, -31, -23, -14, -33, -24, -32, -11, -22, -16, -20, -19, -23, 1, -10, -1, -16, -2, -23, -6, 36, 15, -12, -11, -1, -5, -24, -16, -39, -4, -3, 6, 20, 19, 15, 23, 17, 8, 3, -10, 4, -3, -7, -21, -9, -27, -32, -20, -17, 19, 7, -10, 7, -19, 5, -3, -2, 24, 24, 14, 32, 19, 27, 28, 22, 27, 11, 22, 21, 18, 1, 7, -4, -12, -6, 2, -15, 11, -13, -19, 4, -19, 27, 20, -7, 8, 21, 18, 35, 35, 43, 38, 19, 35, 21, 23, 24, 24, 16, 18, 2, 20, 19, 22, 28, 16, -2, -13, -12, 12, 17, 5, -4, 11, 32, 31, 38, 37, 30, 23, 20, 13, 21, 12, 20, 17, 20, 12, 13, 17, 12, 48, 66, 54, -9, -12, 14, 27, 0, 5, -20, 16, 21, 21, 23, 20, 34, 32, 27, 17, -1, -12, 2, 25, 18, 25, 12, 8, 20, 22, 47, 35, -11, 4, 0, 16, 15, -10, 7, 24, 16, 26, 25, 45, 24, 20, 28, -13, -19, 3, 9, 8, 26, 19, 15, 22, 21, -2, 8, 16, 6, 6, 15, 19, 21, -10, 5, 11, 31, 33, 28, 36, 30, 23, 0, -26, -18, 5, 19, 24, 18, 15, 5, 10, 11, -21, -29, 3, 20, 2, 13, 35, 52, 8, 18, 14, 12, 22, 33, 24, 14, 19, -20, -25, -35, 8, 14, 21, 18, 22, 20, 32, 10, -24, -36, -6, 39, -5, 0, 31, 42, 17, 31, 14, 6, 33, 18, 12, 30, 24, -25, -24, -22, 2, 15, 10, 13, 24, 27, 30, 25, -16, -21, -1, 49, -16, 6, 8, 21, 18, 57, 4, 4, 30, 23, 20, 23, 16, -12, -20, -17, -6, 8, 35, 31, 34, 30, 13, 12, 0, -10, 1, 26, -15, 14, -14, -13, 15, 23, 3, 13, 22, 28, 28, 22, 2, -7, -7, -27, 9, 19, 57, 45, 29, 2, 10, 8, -12, -22, -4, 38, 33, 22, -27, -29, -6, -2, -10, -3, 31, 28, 5, -2, -1, 14, -9, -9, 34, 38, 56, 15, 2, 10, -3, -5, -14, -30, -6, 28, 16, 2, -21, -14, -25, -14, -1, 12, 10, 6, -3, -3, 20, 16, 20, 8, 34, 36, 28, 11, -3, 9, 2, -13, -26, -12, 10, 17, -10, 9, -23, 4, -32, -10, 0, 6, 0, -21, -14, -6, 9, -2, 7, 20, 24, 17, -7, -5, -6, 1, -13, -17, -18, -5, -18, -3, -17, -5, 18, 12, -26, -1, -3, -1, -12, -7, -32, -20, -20, 0, -2, 9, -5, -18, -8, -10, -7, -13, -23, -17, -14, -6, -21, 6, -14, 7, 12, 9, -29, -20, 0, -8, -17, -12, -22, -27, -20, -13, -1, -7, -9, -13, -13, -21, -16, -18, -8, -12, -18, 12, 1, 1, -12, -11, -19, -3, -35, -21, -18, -25, -21, -10, 1, -26, -8, 1, -12, -15, -6, -8, -17, -25, -20, -16, -5, -16, 10, 31, -4, 3, 3, -15, -23, -7, -19, -18, -28, -21, -15, 9, 6, 8, 6, 18, 15, 13, -6, -5, -13, -27, -21, -16, 8, 3, 19, 28, 7, -18, -6, 9, 7, -25, -20, -9, -37, -19, -7, 4, -3, 7, 6, 16, 2, 11, 3, 9, -11, -8, -12, 4, 22, 29, 30, 16, 13, -13, 10, -4, 9, -5, -17, 1, 4, -6, -10, 12, -4, 3, 18, 19, 16, 13, 29, 16, 8, 2, 5, 20, 18, 24, 1, -5, -20, -19, -5, 6, -1, -20, 18, 15, 18, -5, -4, 2, 5, 20, 20, 9, 27, 24, 33, 21, 20, 20, -5, 4, 22, 4, -8, -14, -9, -9, -2, -1, -5, -2, 16, -2, -2, 2, 13, 12, 16, 21, 18, 20, 28, 37, 41, 23, 20, 10, 16, -1, 17, 6, 9, 4, 12, -6, -9, -6, -5, -4, 3, 13, 6, -5, -5, 12, 17, 19, 10, -3, 24, 27, 22, 21, 29, 32, 39, 14, 24, 33, 16, 7, -4, -6, -1},
{-3, 4, 0, 10, -5, -7, -2, 6, 5, 6, -9, -4, 15, 6, -1, 7, 4, -6, -9, -5, -3, -1, -2, -3, 9, 8, 8, -7, 6, 7, -9, -5, 13, 10, 20, 30, 15, 28, 44, 12, 10, 8, 3, 0, -7, 20, 35, 38, 20, 16, 17, 8, 8, -5, -9, -7, 5, -5, 1, -9, 5, 20, 24, 32, 26, 22, 35, 33, 30, 36, 38, 49, 28, 16, 9, 22, 35, 36, 30, 34, 28, 16, -8, -8, 9, -8, 23, 6, 8, 8, 22, 31, 21, 19, 12, 19, 19, 14, 25, 22, 19, 16, 18, 30, 12, 9, 13, 0, -8, 2, 5, 5, 3, 3, 17, 39, 51, 32, 19, 4, 12, -8, 1, 2, -3, 8, -2, -5, -16, -12, -3, -8, -20, -25, -34, -27, -13, -6, -12, -7, 1, -7, 21, 10, 34, 36, 10, 15, 5, -1, 3, -3, 9, 5, -4, 5, -14, -34, -22, -15, -24, -26, -49, -50, -55, -49, -23, -4, -6, 2, 32, 41, 3, -9, -11, -10, 4, 8, 12, 18, 5, 17, 22, 7, 20, 7, 5, -11, -13, -20, -42, -69, -64, -49, -36, -13, 3, 12, 20, 27, 3, -22, -16, -4, -7, -4, -2, 11, 19, 16, 19, 23, 12, 14, 9, 4, -1, -23, -34, -53, -70, -37, 9, -32, -8, 24, 17, -4, 11, -10, -12, -2, 3, -10, 4, -1, 3, 15, 29, 18, 8, 8, -6, -4, -18, -10, -13, -48, -65, -50, -17, -25, 3, 36, 4, 4, 18, -11, 8, 12, 4, -2, 0, -5, 6, 1, 30, 42, 5, 4, 0, -16, -10, -14, -12, -47, -63, -49, -42, 4, 6, 25, 15, 9, -11, 3, 12, 16, -1, -7, -8, -1, -18, 4, 29, 34, 25, -4, -9, -2, -6, -8, -2, -35, -58, -54, -36, -3, 15, 4, 34, 29, 4, 8, -1, -1, 10, 5, 0, 5, -2, 10, 27, 31, 29, 22, 15, -1, 27, 12, 18, -18, -50, -59, -53, 3, 18, 4, 33, 8, 28, 23, 11, 22, 10, 8, 10, -2, -21, 2, 43, 42, 29, 33, 29, 9, 16, 21, 11, 18, -19, -41, -54, -3, -6, -5, 12, 2, 38, 12, 18, 15, 3, -1, -8, -24, -30, -2, 32, 32, 40, 21, 9, 9, -2, 20, 8, -6, -18, -39, -65, -20, -8, 1, 11, 18, 7, -7, 4, -6, -10, -1, -20, -13, -25, -6, 25, 22, 28, 25, 14, 9, 20, 10, 8, -7, -38, -32, -54, -17, -7, 17, -16, -3, -9, -20, -7, -14, -6, -23, 1, 5, -2, -3, 20, 26, 18, 23, 15, 12, 25, 12, 11, -10, -32, -39, -49, -14, 14, 11, 9, -12, -1, -6, -5, -1, -3, -3, 9, 9, -7, 0, 12, 14, 5, 0, -2, 2, -2, -2, -13, -34, -47, -50, -53, -26, -9, 28, -14, -16, -19, -26, -31, -19, 0, -13, -1, 0, -7, 3, 22, 17, 4, -3, 7, 11, 1, 0, -6, -11, -53, -53, -40, 0, 21, 29, 12, -6, -27, -42, -37, -27, -26, -13, -9, 3, -1, 3, 20, 2, 9, 12, -2, 1, 1, -7, -2, -24, -80, -41, 9, 11, 5, 13, -7, -14, -23, -21, -32, -28, -21, -20, -26, -2, -7, 10, 9, 1, 5, -7, 5, 14, -1, -8, -22, -30, -48, -11, -3, 25, -4, -15, -28, 0, -6, -19, -17, -13, -26, -13, -20, -5, -12, 6, 17, -5, 11, -1, 4, -8, -2, -25, -30, -35, -14, 1, 16, 21, 20, -1, -14, -3, -10, -33, -12, -10, -1, -6, -9, -16, -3, -9, 4, 5, 5, 0, -7, -8, -17, -18, -41, -40, -16, 8, -26, 3, 13, 15, -17, -19, -6, -10, 3, -7, -3, -1, 10, -12, -2, -17, 2, -5, 3, -4, -17, -8, -16, -31, -22, -34, -11, 8, -15, -2, -8, 5, -6, 3, 15, 18, 18, 15, -10, -6, -4, -20, -16, -14, -24, -10, -18, -13, -16, -26, -21, -35, -16, -12, 4, -16, 16, -1, 5, -1, 4, 21, 48, 58, 38, 25, 5, -10, -1, -3, -11, -18, -18, -11, -17, -10, -19, -29, 2, -5, -7, 10, -27, -15, 9, 7, 7, 9, 9, -1, 41, 65, 40, 54, 58, 57, 20, 22, 35, 7, 16, -16, -2, -5, 4, 21, 23, 23, -7, 8, -4, 5, -1, -1, 6, -10, 1, 9, 18, 34, 60, 82, 84, 87, 70, 55, 43, 66, 89, 69, 83, 70, 62, 56, 31, 35, 25, 6, 5, -4, 0, 2, -1, -5, 8, 6, 6, -2, 12, 17, 28, 30, 41, 6, 27, 36, 77, 52, 53, 59, 37, 47, 22, 36, 30, 5, -10, 1, -5, 3},
{5, -5, 4, 1, 8, -3, -6, -7, 5, 2, 3, 4, 1, 5, 2, -4, -6, -4, 4, 4, -7, 8, -2, -8, -8, 8, -5, 5, 4, 1, -3, -6, 1, 22, 17, 16, 19, 43, 45, 32, 42, 40, -16, -14, 21, 26, 48, 36, 35, 19, 29, 18, 2, 3, 7, -9, 2, 7, -3, 19, 22, 27, 39, 55, 40, 32, 74, 68, 78, 54, 15, -5, -20, 12, 8, 10, 8, 26, 19, 28, -5, -11, 8, -4, 5, 9, -19, 25, -10, 18, 44, 47, 57, 39, 48, 72, 38, 48, 32, 11, 7, 28, 14, 2, -1, -3, 10, 11, -2, -2, -6, 5, 6, 15, 2, 7, 26, 29, 21, 18, 23, 36, 37, 36, 24, 29, 22, 4, 19, 13, 9, -6, 5, -13, -16, -12, -37, -8, 17, 27, 7, 7, 0, 3, 26, 0, 14, 7, 20, 25, 19, 21, 36, 30, 19, 24, 28, 22, 15, 13, -3, -16, -30, -35, -23, -14, -1, 13, 2, 17, -3, 22, 18, 4, 17, 17, 8, 16, 17, 10, 14, 28, 18, 27, 26, 22, 5, -13, -2, -4, -22, -13, -31, -19, -25, -5, -8, 13, -4, 17, 57, 27, 29, 3, -5, 11, -1, 5, 5, 14, 9, 26, 7, -1, 4, -10, -16, -6, -10, -14, -22, -34, -10, 0, 20, -16, -12, 30, 59, 32, 32, 21, 2, 13, 5, -4, 9, 14, 14, 21, 22, 6, -1, 5, -4, -13, 6, -21, -25, -40, -9, 5, -1, -33, -9, 25, 56, 37, 12, 8, -4, 8, -6, -16, -12, 14, 16, 11, 18, 24, 16, 19, 17, -7, 1, 0, -64, -57, -30, -28, -10, -15, -12, 40, 37, 16, 15, -12, -18, -27, -18, -21, -26, -5, -11, -19, -13, 20, 31, 29, 38, 14, 18, 8, -61, -101, -76, -21, -2, -6, -24, 36, 1, 0, -12, -20, -17, -10, -7, -19, -22, -15, -18, -37, -23, 7, 18, 36, 46, 48, 50, 31, -8, -91, -56, -6, -7, -18, -32, 5, -19, -18, -20, -15, -16, 5, 0, -11, -14, -15, -32, -27, -16, -3, 6, 28, 23, 40, 78, 107, 127, 15, -37, 9, 0, -19, -33, -12, -31, -33, -1, -16, 4, 3, -2, -1, -12, 0, -26, -21, -7, 3, 19, 7, -3, 7, 30, 67, 102, 53, -2, 34, -11, -6, -29, -24, -47, -2, -1, 18, 8, 7, 7, 10, 4, -9, -7, -12, -9, 6, 15, 16, -8, 4, 10, 6, 18, 33, 12, 24, -6, 1, 2, -19, -25, 16, 15, 19, 19, 15, 16, 0, -6, 12, -10, 1, -2, 15, 6, 12, 12, 4, 19, 0, -6, 29, 14, 17, 18, 16, 40, 8, -3, 24, 12, 38, 27, 28, 16, 14, 15, 3, -7, -13, 7, 23, 22, 5, 5, -1, -4, 3, 2, 66, 38, 7, 9, 9, 35, 14, -5, -3, -4, 22, 32, 32, 32, 36, 18, -4, -18, -13, 19, 11, 8, 13, 10, 13, 9, 7, 5, 68, 22, -7, -20, 10, 5, 13, -1, 4, 2, 12, 14, 19, 20, 15, 11, -8, -25, 4, 31, 22, 29, 7, 0, 19, 21, 19, -3, 44, 6, 9, -16, -17, -20, 7, -5, 23, 11, 10, 2, 5, 25, 10, 7, -7, -6, 13, 9, 22, 10, 4, 0, 25, 24, 14, -12, 8, 6, -6, -4, 4, -11, 8, 11, 31, 21, 21, 6, 6, 18, 4, 19, 19, 16, 20, -1, 8, 12, 13, 15, 18, 1, 12, -5, -23, -12, -18, -5, 26, 33, 20, 7, 6, 8, 14, 3, 7, 3, 7, 19, 21, 10, 13, 12, 0, 15, 17, -1, -8, 3, 13, -12, -5, 13, 3, -7, -4, 46, 22, -12, -2, 5, 9, 6, 8, 8, 23, 24, 29, 13, 10, 0, 7, -5, -12, 5, 7, 1, -13, -23, 2, -25, -2, -2, -3, 26, 13, -13, -30, -12, 6, 6, 5, 4, 17, 9, 22, 17, 15, -10, -6, 25, 0, 9, 19, 7, 4, -17, -7, -16, -6, 5, -8, 14, -3, 16, -25, -17, 0, 2, -7, 14, 11, 10, 14, 9, 9, 22, 3, -12, -22, -22, -13, -17, -8, -8, 21, 3, 6, 10, -9, 17, -15, -13, 21, 29, 25, 25, -7, 29, 1, 14, 3, 9, 5, -10, -19, -21, -12, -24, -42, -41, -36, -36, 8, 10, 2, -9, 4, -7, -14, 33, 10, 18, 35, 16, 8, 11, -11, -13, -2, 4, -22, -31, -4, 0, -20, -11, -12, -25, 0, 17, -1, 0, -8, -8, 9, 4, -3, -26, -8, -7, 5, -1, 5, -38, -16, -16, -43, 0, -9, -27, -13, 8, -7, -2, -2, -24, -17, 8, 2, 4, 4},
{-8, -4, -8, 2, -7, 6, 7, -4, 7, 4, 3, 5, -3, -5, 4, 4, -7, -4, 0, -5, -7, 8, -3, -8, -10, -9, -4, -9, -6, -5, -1, 1, 2, -9, -14, -15, -8, -11, -13, -29, -36, -24, -9, -4, 43, 24, -12, -17, -3, -9, -4, -3, 3, 2, -8, -1, -7, -4, 9, 2, -17, -29, -17, -19, -21, -1, -1, 8, 7, 12, 7, 11, 15, 1, -2, 1, -1, -14, -12, -13, 7, 14, 5, -6, 8, 5, 12, 8, -3, 13, 11, 15, 25, 24, 24, 45, 46, 45, 34, 29, 26, 22, 6, 4, 0, -10, -24, -43, -7, -1, -13, 4, -2, 7, 11, 19, 7, 28, 37, 42, 31, 45, 34, 41, 47, 33, 30, 20, 12, 11, 14, 8, 0, -29, -28, -44, -16, -16, -9, -23, 5, 1, 31, 39, 24, 28, 30, 42, 19, 20, 18, 29, 25, 30, 15, 24, 27, 12, 10, -2, -12, -6, -2, -27, -17, -27, -9, -12, 8, 18, 14, 29, 20, 18, 36, 16, 12, 20, 39, 27, 15, 32, 28, 19, 23, -1, 17, -6, -5, 0, -8, -11, -32, -41, -17, -11, -3, 24, 7, 33, 30, 14, 23, 9, 13, 21, 25, 25, 30, 36, 41, 16, 22, 13, 2, 19, 4, 1, 7, -4, -57, -25, -20, -9, -5, 8, -9, 6, 35, 17, 0, 8, 11, 15, 21, 17, 14, 19, 30, 29, 11, 13, 13, 18, 10, 1, 19, -4, -56, -16, -24, 0, -6, 7, 15, 23, 26, 24, 10, 15, -2, 10, -7, -21, -19, -11, 5, 29, 30, 24, 30, 17, 11, 9, 12, -6, -45, -14, -13, 10, 8, 2, 13, 29, -8, 6, 6, -21, -32, -44, -62, -47, -56, -41, -11, 13, 19, 19, 0, 10, 9, -8, 1, -15, -41, 25, -9, 22, 3, -10, 7, 0, -35, -37, -41, -74, -68, -74, -56, -51, -58, -46, -26, 2, 1, 9, 13, 12, -5, -18, -31, -47, -19, 19, -11, 10, 4, -8, -23, -30, -48, -70, -87, -82, -52, -44, -27, -23, -27, -37, -1, 1, 0, -6, 3, 5, -4, -38, -42, -60, -26, 25, 4, 20, -4, -10, -10, -24, -56, -81, -78, -56, -19, -19, -13, -8, -3, 13, 16, -11, -2, 2, 7, 3, -13, -26, -33, -35, -13, 28, 21, 8, 5, -9, -9, -7, -23, -49, -43, -16, -5, 0, -2, -7, 8, 19, 26, -7, -7, 5, 5, -8, -17, -12, -25, -22, -6, 2, 37, 11, 2, -1, -5, 8, 19, 11, 4, -18, -21, 3, -1, -6, -4, 8, 11, 1, -2, 5, -6, -2, -9, -2, -8, -10, 12, 16, 38, 28, -5, 17, 22, 38, 56, 24, 15, -14, -32, -8, -4, -7, 3, 20, 3, -9, -6, 5, -4, 3, 2, 2, 11, 8, 23, 25, 48, 32, 4, 25, 11, 34, 58, 38, 23, -15, -27, -29, -25, -26, -23, -4, -8, -1, 0, -7, 5, 10, -3, 2, 5, 12, 13, 33, 45, 33, 13, 24, 7, 42, 54, 34, 22, 6, 0, -24, -18, -31, -21, -10, -5, 3, 9, 18, 12, 16, 9, 29, 25, 29, 31, 31, 22, 34, 5, 21, -13, 31, 46, 32, 26, 19, 16, 10, 2, -6, -13, 5, 8, 3, 11, 16, 7, 20, 15, 25, 18, 18, 23, 49, 40, 22, 1, 16, -12, 45, 41, 26, 37, 28, 21, 19, 8, 4, -22, -15, -8, -3, 9, 0, 17, 14, 15, 16, 11, 23, 13, 40, 35, 2, 11, 25, 52, 50, 22, 16, 28, 15, 9, 9, -3, 3, -10, -16, -2, 1, 6, 10, 7, 26, 11, 14, 13, 24, 25, 36, 20, -5, -1, 14, 55, 40, 13, 20, 21, 18, 8, 17, 7, -1, -12, -13, -6, -7, 0, 11, 9, 29, 21, 23, 34, 27, 43, 8, -5, 2, -3, 7, 36, 29, 26, 33, 25, 23, 10, 23, 10, 22, 7, 7, -12, 4, -8, -5, 17, 23, 21, 10, 30, 42, 51, 14, 15, 8, 10, 5, 12, 4, 35, 15, 15, 21, 7, 22, 13, 12, 24, 16, 17, 14, 6, 22, 1, -9, -1, 27, 27, 27, 53, 44, 11, 9, 7, 4, 8, -16, -8, 4, 26, 9, 38, 25, 21, 23, 43, 12, 8, -1, 5, 7, 7, -5, -4, 3, 6, 2, 24, 23, 11, -6, -5, 3, -6, -14, -6, -18, -19, 8, 17, 11, 10, 4, 14, 27, 28, 43, 34, 23, 23, 28, 14, -5, -17, -7, -5, -1, -10, 7, 4, -1, 6, -4, 6, 12, 4, 11, 8, -12, 14, 17, 13, 21, 21, 0, -4, 5, 3, -13, -20, -14, -16, -4, -9, 4, 6, -1},
{-1, -8, 1, -7, -9, -3, 4, -7, 5, 6, -4, 2, -5, -3, 1, 7, -4, 7, 7, -6, -6, 6, -3, -2, 9, -7, -3, 3, 5, 4, 5, -3, 3, -1, 10, -4, 10, -6, -21, -32, -30, -11, 17, 37, 31, 10, -2, -2, -15, -14, -14, -8, -7, -6, 5, -9, -6, 4, -5, -9, -7, -10, 10, -1, -8, -18, -12, 3, 5, -22, 15, 37, 31, -10, -23, -45, -40, -37, -25, -16, 3, 23, 7, -2, -7, 1, 23, -8, -11, -10, -9, -2, -26, -19, -18, -27, -15, -6, -13, -7, -8, -27, -28, -46, -41, -41, -27, -3, 3, -7, -11, 0, -9, -12, 28, -17, -4, -21, -18, -20, 1, 1, 2, 3, 31, 27, 32, 32, 12, -5, 4, -2, -5, -18, -30, -26, -29, -15, 14, -13, -9, -4, 28, 0, 17, 1, -9, -5, -22, -13, -20, 11, 17, 20, 15, 30, 13, -2, -6, -5, -3, -5, -12, -27, -32, -23, -22, -15, -7, 1, -8, -5, 21, -2, -18, -8, -7, -13, 4, 14, 18, 10, 13, 8, 4, -2, 0, -1, -9, -2, -10, -4, -36, -1, -8, -16, -6, 5, -4, 9, 14, -6, -8, -3, -2, 5, 8, 9, 9, 17, -3, 11, -2, 0, 12, 10, 8, 8, -6, 3, -25, -25, 2, -23, -5, 17, 4, 16, 20, 4, 3, 13, 7, 14, 12, 16, 23, 20, 11, 3, 21, 19, 10, 22, -6, -5, 5, -9, -16, -11, 9, 10, 9, 21, -20, -3, -10, 5, 2, 4, 9, 13, -4, 9, -6, 5, 8, 15, 7, 12, 18, 5, 8, 8, 7, -17, -37, -23, 9, -25, -7, 19, -7, -32, -4, -13, 7, 23, 1, 5, -6, -11, -22, -20, 18, 13, 10, 6, 13, 13, 7, 18, 2, -10, -53, -55, -26, -4, -7, 5, -7, -46, -29, 12, 11, 29, 17, 5, -13, -28, -27, -9, 36, 26, -3, -10, -9, 9, 8, 14, 34, 4, 0, -24, -10, 11, -2, 5, -17, -19, -19, 19, -1, 13, -6, -21, -33, -35, -19, 12, 29, 19, 3, 2, 2, 7, 16, 46, 40, 45, 58, 34, 5, 7, 0, 8, -12, -19, -40, -12, -29, -29, -30, -21, -40, -18, -7, 10, 33, 26, 18, -4, -6, -14, 16, 3, 11, 32, 41, 56, 40, 33, 1, -14, -10, -8, -32, -39, -50, -59, -64, -34, -27, 12, 15, 27, 24, 12, -6, -12, -5, 10, -14, -3, -1, 4, 18, 39, 70, 22, -2, -1, 1, -8, -25, -44, -62, -39, -33, -14, -2, 16, 7, 25, 36, 6, -9, -5, -9, -21, -24, -21, -21, -8, 5, 17, 25, 30, -1, -10, -15, -12, -60, -62, -32, -4, 11, 18, 24, 17, 17, 28, 24, -9, -3, -21, -27, -24, -17, -25, -30, 5, 4, 6, 26, 36, -3, -15, -26, -44, -53, -73, -45, 10, 20, 28, 33, 24, 26, 21, 15, -15, -23, -26, -34, -17, -21, -30, -10, 0, 5, -5, 29, 31, -6, 8, -15, -37, -59, -58, -40, 0, 17, 40, 31, 42, 21, 17, -4, -24, -28, -30, -23, -14, -8, -18, -6, -1, -13, -15, 8, 26, 6, 17, -26, -23, -31, -45, -25, 0, 6, 35, 34, 12, 7, 4, -13, -23, -14, -22, -14, -8, -10, -16, -1, 8, -11, -10, 26, 36, 2, 4, -15, -16, -22, -30, -4, 17, 5, 8, 21, 18, 11, 5, -5, -4, -10, -4, -18, -7, 0, -1, 3, -3, -9, -21, 18, 6, 6, 16, 0, -23, -20, -7, 10, 13, 3, 15, 11, -1, 3, 8, 0, 7, -2, 2, -3, -5, 8, 1, 14, 8, -27, 8, 8, -4, 16, 2, -9, -27, -7, 32, 19, 22, -2, 5, 2, -1, -1, 2, 24, 5, 9, -5, -7, 4, -9, 3, 9, 8, -11, -19, -5, 5, 8, 4, -17, -41, -34, -4, 2, 8, 13, 10, 0, 21, 19, 31, 11, 24, 20, 13, -3, 4, -17, -7, 4, -1, -24, -22, 4, 2, -6, 1, 2, -31, -51, -37, -15, -11, -7, -2, -2, 3, 9, 8, 15, 7, 0, -9, 10, 15, 1, -14, -22, -17, -10, 14, -10, 9, -6, -8, -11, 15, -36, -59, -38, -40, -21, -34, -18, -15, -21, -25, -23, -5, -16, -3, -10, -9, -19, -16, -35, -28, -35, -30, -16, 4, 6, -5, 7, -5, -49, -41, -49, -23, -12, -35, -47, -43, -54, -55, -39, -51, -63, -53, -40, -57, -51, -34, -26, 15, -3, 1, 0, 2, 0, -8, 8, -6, 24, 16, 25, 24, 21, 2, 23, 22, -13, -11, -8, 10, 8, 2, -18, -9, -16, -3, -5, -5, -8, -10, -2, 0},
{-7, -8, 4, 7, -5, -7, 3, 7, -9, -10, 7, 5, -11, -15, 0, -3, 3, 5, -3, 7, -9, 5, -6, 3, 4, 8, 7, 9, -6, -2, -7, -6, 3, -19, -30, -30, -13, -1, 2, -13, -22, -15, 18, -14, -33, -21, 19, 0, -14, -18, -13, -17, -10, 6, 5, 5, 8, 0, 4, -7, -3, 1, 1, -15, -28, -27, -32, -15, -23, -18, 14, 14, 7, 16, 12, 32, 27, 14, 14, -9, -9, -21, 8, -3, -1, -8, -2, -21, 8, 18, 0, 8, -15, -40, -9, -21, -18, -27, -24, 7, -13, 6, 25, 33, 41, 45, 45, 52, -5, -5, -2, -1, -4, 3, 25, 7, 23, 28, 1, 13, 0, -11, 15, 13, 27, 17, 22, 7, 25, 25, 28, 20, 33, 20, 22, 26, 53, 22, 0, -21, 5, -8, 10, 11, 25, 23, 5, 3, -1, 17, 2, 8, 19, 8, 11, 20, 18, 6, -1, 17, 18, 2, -5, 22, 36, 45, 9, 7, 8, 25, -11, -3, -1, 13, 2, -1, 3, 17, -1, -2, 2, -7, -4, 1, -8, 2, -15, 0, -8, -18, -12, 37, 55, 79, 56, 22, -4, 20, -26, -25, 5, 18, -4, -1, 23, 16, 9, -4, -9, -10, -6, -3, -14, -13, -14, -2, -11, 0, -5, 19, 39, 53, 67, 7, -18, -41, -19, -9, -2, 17, -1, 9, -6, 3, -1, -1, -5, 0, -11, -25, -19, -14, -4, -13, -6, 9, 7, -9, 40, 84, 52, 27, -19, -21, -13, -10, 2, -4, 18, 17, 13, 9, 0, -1, -9, -16, -28, -29, -15, -19, -16, -8, -2, 6, 20, 12, 20, 66, 46, -12, -5, -10, -26, -5, 9, -19, -6, 7, 7, 5, 2, -6, 11, 0, -25, -8, -23, -18, -19, -14, -10, -13, -7, -13, 25, 50, 41, 0, -4, -19, -34, -15, 2, -9, 2, 3, 14, 15, 25, 18, 35, 17, 18, 21, 8, 1, -8, -13, -23, -33, -15, -16, 6, 23, 54, -2, -16, -12, -29, -32, -12, 4, 19, 21, 9, 24, 26, 26, 52, 47, 43, 30, 27, 8, 0, -1, -7, -4, -26, -10, -8, 0, -7, -25, 0, -7, -31, -24, -9, 15, 8, 4, 11, 12, 18, 25, 39, 46, 33, 42, 32, 23, 10, 17, 11, -7, -16, -5, -19, -29, -51, -22, 6, 3, -17, 2, 3, -3, 0, -2, 5, 19, 7, 21, 23, 36, 49, 49, 32, 33, 14, 1, -11, -11, -11, -23, -22, -29, -73, -24, -9, 11, 30, 12, -7, -19, -14, -11, 5, 10, 0, 26, 43, 47, 56, 44, 39, 16, 5, 1, -5, -7, -7, -4, -36, -41, -28, -36, -14, 21, 7, 8, -34, -21, -20, -17, -14, -4, -4, 14, 34, 43, 49, 39, 27, 2, 14, -2, -2, -16, 0, 5, -8, -50, -60, -21, -3, 26, 21, 12, -8, -15, -7, -14, -17, -21, -20, -14, 7, 23, 23, 19, 28, 13, 3, 8, -2, 12, -8, 5, -9, -88, -46, -32, -10, 20, 20, 32, -15, 5, 1, -12, -7, -21, -24, -8, 12, 19, 17, 14, 21, 12, 2, 11, 9, 0, 1, 5, -7, -59, -17, -15, 18, -26, -8, 18, -8, 15, 9, -5, -16, -7, -20, -13, -14, 3, 4, 22, 22, 14, 27, 18, 0, 8, 2, 8, -10, -42, -21, -2, -4, -23, -12, 3, -14, -10, 10, 7, 0, -12, -2, -6, -17, -5, 18, 19, 16, 24, 28, 18, 12, 19, 16, 22, -13, -34, -5, 4, 11, 9, 14, -6, -11, -9, 0, -1, 13, 7, -6, 1, -2, 2, 9, 26, 10, 25, 24, 22, 18, 13, 13, -8, -34, -31, -16, -4, 11, 10, 11, 9, 18, -20, -16, -8, -12, -4, -8, 5, 10, 6, 1, 9, 13, 14, 15, 17, 13, 26, -6, -5, -18, -15, 12, -1, -6, 2, 34, 44, 34, -4, -15, -11, -6, 4, -5, 4, 3, 8, 5, 8, 28, 14, 33, 17, 15, 13, -30, 0, -4, 2, 30, -8, 5, -1, -4, 25, 13, -11, 9, 9, 21, 31, 32, 31, 13, 15, 16, 14, 17, 13, 13, 26, 9, 1, 5, 18, -39, -18, -21, -1, 3, -8, 23, -19, -25, -17, -1, 21, 14, 23, 25, 22, 27, 24, 24, 15, 12, 1, 10, 13, 20, 7, 7, 34, -2, -15, -17, 3, -6, 8, -5, -8, 16, -7, -8, -22, -11, -24, -26, -25, -25, -36, -45, -35, -15, -23, -25, -32, -20, -21, -26, 10, 11, -14, -5, -1, -5, -5, -1, 8, -27, -22, -12, -28, -29, -23, -40, -12, -14, -52, -35, -26, -14, -48, -21, -42, -5, -18, -25, -15, -2, 1, 9, -4}
};

const int layer1_b[10] = {-2943, 11454, 4782, 11098, 10803, 6071, -14899, 3927, -13007, -1090};

const char layer2_w[10][10] = {
{-87, 40, 34, -21, 23, -27, 36, -25, -5, -62},
{39, -14, -26, 3, -72, 34, -85, 26, 76, 32},
{55, -21, -21, 50, 9, -46, 27, 58, 27, -25},
{-1, 33, -13, -71, -31, 24, 32, 64, -18, 41},
{-93, -65, -63, -13, 51, -32, 12, -63, -4, 67},
{-10, 48, 40, -9, 19, -59, -55, 9, -49, 22},
{-45, -1, -2, 56, -37, 17, 47, -127, -13, 15},
{17, 28, -69, -42, 54, 51, -7, 27, 42, -39},
{18, -25, 6, -36, 13, -49, 25, -23, 43, 49},
{-27, -87, 25, -77, 45, 56, 12, -25, -61, 4}
};

const int layer2_b[10] = {-96, 329, -100, -304, 383, 776, 71, 263, -622, 69};

