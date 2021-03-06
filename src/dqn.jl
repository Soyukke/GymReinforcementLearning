using Flux
using Flux.Data
using Random
import Flux:@functor, onehotbatch, @nograd
import Statistics.mean, Statistics.std
import Base.Iterators.partition
export DQNetwork, loss, run_episode, dqnmain, huber_loss
export argmaxs, nomax_novalue

include("replay.jl")

"""
DQNネットワーク
"""
struct DQNetwork
    nstatetype::Integer
    naction::Integer
    env::Environment
    model 
end
@functor DQNetwork

"""
Deep Q-Networkインスタンス作成
DQNモデル
s -> Q(s, a1), Q(s, a2), Q(s, a3)
## 状態ベクトルの次元数
nstate_type
## 行動の種類
nation
"""
function DQNetwork(env::Environment)
    obs = observation_space(env)
    action = action_space(env)
    nstate_type = n_state_features(obs)
    naction = n_action(action)
    @show nstate_type, naction

    model = Chain(
        Dense(nstate_type, 32, relu),
        Dense(32, 128, relu),
        Dense(128, naction)
    )
    return DQNetwork(nstate_type, naction, env, model)
end

function qs(q::DQNetwork, s::Vector{T}) where T <: Number
    s = Float32.(s)
    predicts = q.model(s)
    return predicts
end

"""
行列`m`の縦方向ごとに見て，argmaxを計算する
"""
function argmaxs(m::Matrix{T})::Vector{Integer} where T <: Number
    return mapslices(argmax, m, dims=1)[1, :]
end

"""
run_episode(dqn::DQNetwork; is_render=false, ϵ=0.3)

1エピソード実行する
ペアでデータを保持しておく: s_i, a_i, R(s_i), s_{i+1}
`dqn`のパラメータを使用してϵ-greedy法で実行する．そのときの`ϵ`のデフォルト値が0.3で，
`is_render`が`true`の場合に実行結果をrenderする．デフォルトは`false`
"""
function run_episode(dqn::DQNetwork; is_render=false, ϵ=0.3)
    env = dqn.env
    s0 = Float32.(reset(env))
    data = []
    r0 = 0
    t = 1
    while true
        if is_render
            render(env)
        end
        qs0 = qs(dqn, s0)
        a0 = argmax(qs0)
        # ε-greedy
        a_actual = rand() < ϵ ? rand(range(1, stop=dqn.naction)) : a0
        maxq0 = qs0[a0]
        # 倒立: 0, こけたら以降-1, 195ステップ以降で0が続いていたら1にする
        # 既存設定は倒立状態で+1
        s1, r1, done, info = step(env, a_actual-1)
        # clipping
        if done
            if t ≤ 195
                r0 = Float32(-1)
                # 倒れたらs1の状態0．次状態はない
                s1 = zeros(Float32, dqn.nstatetype)
            # else
            #     r1 = Float32(1)
            end
        elseif 195 ≤ t
            # 195をこえたら報酬1を与える
            r0 = 1
        else
            # たっている
            r0 = Float32(0)
        end
        islimitkey = "TimeLimit.truncated"
        # 既定のステップ数を超えた場合，trueになる
        islimit = haskey(info, islimitkey) && !info[islimitkey]
        # doneか既定のステップ数を超えたら終了する
        qs1 = qs(dqn, s1)
        a1 = argmax(qs1)
        maxq1 = qs1[a1]
        # 学習に必要なパラメータ
        # record = [s0, a_actual, r0, maxq1]
        # 次の状態の最大価値は，都度計算する
        record = [s0, a_actual, r0, s1]
        push!(data, record)

        if done || islimit
            break
        end

        # 更新する
        s0 = s1
        t += 1
    end
    return data
end

function huber_loss(x::Number)
    x = Float32(x)
    δ::Float32 = 1
    if abs(x) ≤ δ
        return 0.5f0 * x^2
    else
        return δ * (abs(x) - 0.5f0 * x)
    end
end

"""
nomax_novalue(m)

Matrix `m`の各列最大値だけ残し，残りを0とする行列をreturnする．
勾配は計算しない

# Examples
入力
```julia
3×2 Array{Float32,2}:
 0.119666  0.344186
 0.837959  0.828465
 0.49077   0.00380257
```

出力
```julia
3×2 Array{Float32,2}:
 0.0       0.0
 0.837959  0.828465
 0.0       0.0
```
"""
function nomax_novalue(m::Matrix{T})::Matrix{T} where T <: Number
    # 行数 = 行動の種類
    naction = size(m, 1)
    @show naction
    action_list = argmaxs(m)
    actions = onehotbatch(action_list, range(1, stop=naction))
    return m .* actions
end
@nograd nomax_novalue

"""
最大値のvectorをreturnする
"""
function maxvalues(m::Matrix{T})::Matrix{T} where T <: Number
    return mapslices(maximum, m, dims=1)
end
@nograd maxvalues

"""
DQNの実行流れ
# Simulation step
1. DQNでsからa ∈ A, Q(s, p)が最大なaをQから予測する
2. aを実行し，T(s, a) -> s'を得る
3. s = s'として1へ戻る．episodeが終了したら終わる

1~3で1エピソードのシミュレーションが終了する

# Prepare step
学習データを作成する
ペアでデータを保持しておく: s_i, a_i, R(s_i), max_p Q(s_{i+1}, p)

# Training step
以下の損失関数を使ってパラメータを学習する．backwardの対象となるのはQ(s_i, a_i)で，max_p Q(s_{i+1}, p)は定数として扱う
model(s_i)の出力をa_iでmaskし，以下の損失関数を計算する
L = (1/2) * (R(s_i) + γ * max_p Q(s_{i+1}, p) - Q(s_i, a_i))^2
"""
function dqnmain()
    gym = Gym()
    env = initenv(gym)
    dqn = DQNetwork(env)
    ps = params(dqn)
    opt = ADAM(1e-3)
    minibatchsize = 500

    """
    loss(s1, a, s2, r1)

    `r1`は状態`s1`における報酬で，`s2`は`s1`で行動`a`を実行するときの遷移後の状態である．
    """
    function loss(s1, a, s2, r1)
        # HACK ここを小さくしたら発散しなくなり，学習がうまくいくようになった．学習率を小さくしてもよいかも
        γ = 0.8f0
        # maxqはnograd
        maxq = maxvalues(dqn.model(s2))
        y = r1 + γ .* maxq
        Y = repeat(y, dqn.naction) .* a
        X = dqn.model(s1).*a
        return L = mean((1f0/2.0f0) * huber_loss.(X - Y))
    end

    try
        isrender = false
        for i in 1:1000
            # replaybufferに10000ステップ分のデータをためる
            replaydata = ReplayBuffer()
            isendreplay = true
            # 平均継続数
            meantime = 0
            numdata = 0
            cnt = 1
            while isendreplay
                # isrender = (cnt == 1)
                data = run_episode(dqn)
                meantime += length(data)
                numdata += 1
                for record in data
                    isendreplay = add_record(replaydata, record)
                    !isendreplay && break
                end
                cnt += 1
            end
            meantime /= numdata

            data = replaydata.data
            @assert length(data) == replaydata.maxsize
            state_list = map(record->record[begin], data)
            states = reduce(hcat, state_list)
            action_list = map(record->record[begin+1], data)
            actions = onehotbatch(action_list, range(1, stop=dqn.naction))
            states2 = Float32.(reduce(hcat, map(record->record[end], data)))
            rewards1 = Float32.(reduce(hcat, map(record->record[end-1], data)))
            traindata = (states, actions, states2, rewards1)
            traindata = DataLoader(traindata, batchsize=minibatchsize, shuffle=true)
            Flux.train!(loss, ps, traindata, opt)
            l = loss(states, actions, states2, rewards1)
            @info i size(states, 2), l, meantime
            # 毎ステップ学習後にϵ = 0で実行して描画する
            data = run_episode(dqn, is_render=true, ϵ=0)
        end
    catch e
        close(env)
        throw(e)
    end
end