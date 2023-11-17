
#ifdef DEBUG3
if ((blockIdx.y == 0) && (threadIdx.y == 0) && (blockIdx.x == 0) &&
    threadIdx.x == 0) {
  print_val("(y,i) - qTile", blockIdx.y + threadIdx.y, i,
            qTile[blockIdx.y + threadIdx.y][i]);
  print_val("(x,i) - kTile", blockIdx.x + threadIdx.x, i,
            kTile[blockIdx.x + threadIdx.x][i]);
}
#endif

// #ifdef DEBUG
//           if ((blockIdx.y == 0) && (blockIdx.x == 0)) {
//             printf(
//                 "sTile[%d][%d]: %f\n", blockIdx.y + threadIdx.y,
//                 blockIdx.x + threadIdx.x,
//                 type2float(
//                     sTile[blockIdx.y + threadIdx.y][blockIdx.x +
//                     threadIdx.x]));
//           }
// #endif