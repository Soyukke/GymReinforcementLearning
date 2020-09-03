module GymReinforcementLearning

using PyCall
using Conda

try
    pyimport("gym")
catch
    println("gymをインストール")
    Conda.add("gym", channel="conda-forge")
end
gym = pyimport("gym")
envs = pyimport("gym.envs")

"""
環境を初期化する
"""
function initenv()
    env = gym.make("MountainCar-v0")
    observation = env.reset()
end

"""
行動を決定する
"""
function decide_action()
end

"""
報酬を評価する
"""
function calculate_reward()
end

"""
行動する
状態が変化する
"""
function execute_action()
end

"""
状態sで行動aを取った後の状態s'が不明な問題の場合
Q(s, a)

## 入力
状態sにおける報酬r, 状態s, 行動a, 遷移後状態s'

## 出力

"""
function Q()
end


end # module
