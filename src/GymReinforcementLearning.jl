module GymReinforcementLearning
export Gym, initenv, example, QLearning
export reset, sample, close, step, render, observation_space
export low, high, n_state_features
export decide_action, execute_action
export main


using PyCall
using Conda

"""
pyimportしたgymのインスタンス
"""
struct Gym
    gym::PyObject
    envs::PyObject
end

"""
シミュレーション環境インスタンス
"""
struct Environment
    env::PyObject
end

struct Observation
    obs::PyObject
end

struct Action
    action::PyObject
end

"""
Q学習構造体
"""
struct QLearning
    env::Environment
    qmat::Array
end

# DQN
include("dqn.jl")

"""
環境情報から行列を初期化する
行列サイズ (離散状態1数, 離散状態2数, ..., 行動数)
"""
function QLearning(env; nstep=10)
    obs = observation_space(env)
    act = action_space(env)
    naction = n_action(act)
    # 状態数はstep^状態ベクトルサイズ
    nfeatures = n_state_features(obs)
    qmat = zeros(repeat([nstep], nfeatures)..., naction)
    return QLearning(env, qmat)
end

function indices(q::QLearning, state, a)
    naction = action_space(q.env) |> n_action
    # 現在の状態sから方策πにより行動aを決定する
    nstep = size(q.qmat)[1]
    obs = observation_space(q.env)
    # 状態の最小値と最大値を使ってnormalizeする
    state_low = low(obs)
    state_high = high(obs)
    nfeatures = n_state_features(obs)
    @assert size(state_low) == size(state_high) == size(state)
    # @info "low, high", state_low, state_high
    state_normal = @. (state - state_low) / (state_high - state_low)
    # 何行目の状態に対応するのか
    # @info "step", 1 / nstep
    # 刻み幅
    step = 1/nstep
    idx_state = ceil.(Int, state_normal / step)
    # 0の場合は0となるので1にする
    idx_state = [max(x, 1) for x in idx_state]
    return [idx_state..., a+1]
end

"""
状態s, 行動aの行動価値を取得する
"""
function (q::QLearning)(state, a)
    idx = indices(q, state, a)
    # s, a
    return q.qmat[idx...]
end

"""
Actionを決定する
"""
function decide_action(q::QLearning, state::Vector)
    naction = action_space(q.env) |> n_action
    # sに対応する行を取得
    qs = [q(state, i) for i in 0:naction-1]
    # ϵ-greedy法
    ϵ = 0.3
    if rand() < ϵ
        return rand(0:naction-1)
    else
        return argmax(qs) - 1
    end

    # greedy
    # return argmax(qs) - 1

    # softmax法
    # @show qs
    # @show exp.(qs)
    # qs = exp.(qs) / sum(exp.(qs))
    # x = rand()
    # @show x, sum(qs)
    # if x <= qs[1]
    #     return 0
    # elseif x <= sum(qs[1:2])
    #     return 1
    # elseif x <= sum(qs)
    #     return 2
    # end
end

"""
行動する
状態が変化する

return s2, r2, done
"""
function execute_action(q::QLearning, s1, r1, action; t=0)
    # 変化前の状態sで行動aをとったときのQ(s, a)を更新する
    s2, r2, done, info = step(q.env, action)
    # 変化後の状態s'からmax_p Q(s', p)を取得して，Q(s, a)を更新する
    # @info s2, reward, done, info
    # Q-Matrixを更新するs
    Q1 = q(s1, action)
    maxQ2 = maximum([q(s2, i) for i in 0:size(q.qmat)[end]-1])
    γ = 0.99
    α = 0.2
    # r1書き換え
    r1 = re_reward(done, t)
    # qmat更新
    q.qmat[indices(q, s1, action)...] += α * (r1 + γ*maxQ2 - Q1)
    return s2, r2, done, info
end

"""
報酬を計算
"""
function re_reward(done::Bool, t::Integer)
    if 195 < t
        return 1
    end
    return done ? -1 : 0
end

function Gym()
    try
        pyimport("gym")
    catch
        println("gymをインストール")
        Conda.add("gym", channel="conda-forge")
    end
    gym = pyimport("gym")
    envs = pyimport("gym.envs")
    return Gym(gym, envs)
end

"""
環境を初期化する
[座標, 速度]
"""
function initenv(s::Gym)
    cartpole = "CartPole-v1"
    mountaincar = "MountainCar-v0"
    env = s.gym.make(cartpole)
    return Environment(env)
end

"""
シミュレーション環境関数
"""
Base.reset(env::Environment) = env.env.reset()
sample(env::Environment) = env.env.action_space.sample()
Base.close(env::Environment) = env.env.close()
step(env::Environment, action) = env.env.step(action)
render(env::Environment) = env.env.render()
observation_space(env::Environment) = Observation(env.env.observation_space)
action_space(env::Environment) = Action(env.env.action_space)

"""
状態変数の定義域
"""
low(obs::Observation)::Vector = obs.obs.low
high(obs::Observation)::Vector = obs.obs.high
n_state_features(obs::Observation)::Integer = first(obs.obs.shape)

"""
Actionの種類
"""
n_action(a::Action) = a.action.n

"""
サンプルしたアクションを実行し，描画し続ける
"""
function example()
    gym = Gym()
    env = initenv(gym)
    reset(env)
    while true
        render(env)
        action = sample(env)
        observation, reward, done, info = step(env, action)
        if done
            break
        end
    end
    close(env)
end

function main()
    gym = Gym()
    env = initenv(gym)
    q = QLearning(env)
    try
        for i in 1:10000
            @info i
            s0 = reset(q.env)
            s = copy(s0)
            r = 0
            t = 0
            while true
                t += 1
                if 3000 < i
                    render(q.env)
                end
                a = decide_action(q, s)
                s, r, done ,info = execute_action(q, s, r, a, t=t)
                if done
                    break
                end
            end
        end
    catch e
        close(q.env)
        throw(e)
    end
    close(env)
end

end # module
