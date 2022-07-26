# def split_xy1(weather_set,time_steps):
#     x,y=list(),list()
#     for i in range(len(weather_set)):
#         end_number=i+time_steps
#         if end_number>len(weather_set)-1:
#             break
#         tmp_x,tmp_y=weather_set[i:end_number],weather_set[end_number]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x),np.array(y)

# x,y=split_xy1(weather_set,3)
# print(x.shape)
# print(y.shape)
