"""
replay buffer
maxsize: 最大記憶サイズ
data: データため込みよう
"""
struct ReplayBuffer
    maxsize::Integer
    data::Vector
end

"""
初期化
空の配列
"""
function ReplayBuffer(;maxsize = 10000)
    return ReplayBuffer(maxsize, [])
end

"""
bufferのmaxsizeになったらfalseをreturnする
"""
function add_record(s::ReplayBuffer, record)::Bool
    push!(s.data, record)
    return size(s.data)[begin] < s.maxsize
end