# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix
import ipdb

class SparseStructure(abc.ABC):
    def __init__(
        self,
        row_ptr: np.ndarray,
        col_ind: np.ndarray,
        value: np.ndarray = None,
        num_rows: int = None,
        num_cols: int = None,
        dtype: np.dtype = torch.float32,  # type: ignore
        device: torch.device = torch.device("cpu"),
    ):
        if row_ptr.device != torch.device("cpu") or col_ind.device != torch.device("cpu") or (value is not None and value.device != torch.device("cpu")):
            device = torch.device("cuda")
            
        mock_col_ind = self.col_ind = col_ind.to(device=device)
        mock_row_ptr = self.row_ptr = row_ptr.to(device=device)
        mock_value = torch.arange(col_ind.shape[-1]).to(device=device)
        self.Bsize = 0
        if len(mock_col_ind.shape) == 2:
            mock_col_ind = mock_col_ind[0]
            mock_row_ptr = mock_row_ptr[0]
            self.Bsize = col_ind.shape[0]
        self.num_rows = row_ptr.shape[-1] - 1 if num_rows is None else num_rows
        size = (self.num_rows, num_cols) if num_cols is not None else None
        if value is not None:
            self.value = value
            if len(value.shape) != len(col_ind.shape):
                self.col_ind = self.col_ind[None].repeat(value.shape[0], 1)
                self.row_ptr = self.row_ptr[None].repeat(value.shape[0], 1)
            if len(value.shape)>1:
                # mock_value = mock_value[0]
                self.Bsize = value.shape[0]
        else:
            self.value = torch.ones(col_ind.shape[-1]).to(device=device)
            if len(col_ind.shape) == 2:
                self.value = self.value[None].repeat(col_ind.shape[0], 1)
        # try:
        self.mock_csr = torch.sparse_csr_tensor(mock_row_ptr, mock_col_ind, mock_value) if size is None else torch.sparse_csr_tensor(mock_row_ptr, mock_col_ind, mock_value, size=size)
        # except:
        #     ipdb.set_trace()
        self.num_cols = num_cols if num_cols is not None else self.mock_csr.size(1)
        self._size = (self.num_rows, self.num_cols) if self.Bsize == 0 else (self.Bsize, self.num_rows, self.num_cols) 
        self.dtype = dtype
        self.device = device
        # if value is not None:
        #     try:
        #         self.csr_tensor = torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=self._size, dtype=dtype)
        #     except:
        #         ipdb.set_trace()

    def transpose(self,) -> 'SparseStructure':
        mock_transpose = self.mock_csr.transpose(0, 1).to_sparse_csr()
        tr_idx = mock_transpose.values()
        # ipdb.set_trace()
        if self.Bsize == 0:
            tr_val = self.value[tr_idx]
            tr_col_ind = mock_transpose.col_indices().to(self.device)
            tr_row_ptr = mock_transpose.crow_indices().to(self.device)
        else:
            tr_val = self.value[:, tr_idx]
            tr_col_ind = mock_transpose.col_indices()[None].repeat(self.Bsize, 1).to(self.device)
            tr_row_ptr = mock_transpose.crow_indices()[None].repeat(self.Bsize, 1).to(self.device)
        return SparseStructure(tr_row_ptr, tr_col_ind, tr_val, self.num_cols, self.num_rows, self.dtype, device=self.device)

    def bmm(self, x: torch.Tensor) -> torch.Tensor:
        if self.Bsize == 0:
            return torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=(self.num_rows, self.num_cols)).mm(x).squeeze(-1)
        else:
            result = [torch.sparse_csr_tensor(self.row_ptr[i], self.col_ind[i], self.value[i], size=(self.num_rows, self.num_cols)).mm(x[i]) for i in range(self.Bsize)]
            # return torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=(self.Bsize, self.num_rows, self.num_cols)).bmm(x)
            return torch.stack(result, dim=0).squeeze(-1)
    
    @property
    def shape(self):
        return np.array(self._size)
    
    def size(self):
        return self._size
    
    def double(self):
        self.value = self.value.double()
        self.dtype = torch.float64
        # self.csr_tensor = torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=self._size, dtype=self.dtype)
        return self

    def cuda(self):
        self.value = self.value.cuda()
        self.row_ptr = self.row_ptr.cuda()
        self.col_ind = self.col_ind.cuda()
        self.mock_csr = self.mock_csr.cuda()
        self.device = torch.device("cuda")
        # self.dtype = torch.float64
        # self.csr_tensor = torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=self._size)#, dtype=self.dtype)
        return self
    
    def to_dense(self):
        return torch.sparse_csr_tensor(self.row_ptr, self.col_ind, self.value, size=self._size).to_dense()
        
    def clone(self):
        return SparseStructure(self.row_ptr.clone(), self.col_ind.clone(), self.value.clone(), self.num_rows, self.num_cols, self.dtype, device=self.device)
    # def csr_straight(self, val: torch.Tensor) -> csr_matrix:
    #     return csr_matrix(
    #         (val, self.col_ind, self.row_ptr),
    #         (self.num_rows, self.num_cols),
    #         dtype=self.dtype,
    #     )

    # def csc_transpose(self, val: torch.Tensor) -> csc_matrix:
    #     return csc_matrix(
    #         (val, self.col_ind, self.row_ptr),
    #         (self.num_cols, self.num_rows),
    #         dtype=self.dtype,
    #     )

    # def mock_csc_transpose(self) -> csc_matrix:
    #     return csc_matrix(
    #         (np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr),
    #         (self.num_cols, self.num_rows),
    #         dtype=self.dtype,
    #     )
