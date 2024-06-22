# 參考了老師提供的資源以及GPT輔助完成
import pulp

linear_problem = pulp.LpProblem("求最大值", pulp.LpMaximize)

x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)
z = pulp.LpVariable('z', lowBound=0)

linear_problem += 3*x + 2*y + 5*z
linear_problem += x + y <= 10
linear_problem += 2*x + z <= 9
linear_problem += y + 2*z <= 11
linear_problem.solve()

print(f'狀態: { pulp.LpStatus[linear_problem.status] }')
print(f'最大值: { pulp.value(linear_problem.objective) }')
print(f'最佳解: x = { pulp.value(x) }, y = { pulp.value(y) }, z = { pulp.value(z) }')