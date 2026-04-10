# chroma查询输出result结构

​    ids: List[IDs]

​    embeddings: Optional[

​        Union[

​            List[Embeddings],

​            List[PyEmbeddings],

​            List[NDArray[Union[np.int32, np.float32]]],

​        ]

​    ]

​    documents: Optional[List[List[Document]]]

​    uris: Optional[List[List[URI]]]

​    data: Optional[List[Loadable]]

​    metadatas: Optional[List[List[Metadata]]]

​    distances: Optional[List[List[float]]]

​    included: Include