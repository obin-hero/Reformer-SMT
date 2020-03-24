import torch
class SensorPack(dict):
    ''' Fun fact, you can slice using np.s_. E.g.
        sensors.at(np.s_[:2])
    '''
    
    def at(self, val):
        return SensorPack({k: v[val] for k, v in self.items()})
    
    def apply(self, lambda_fn):
        return SensorPack({k: lambda_fn(k, v) for k, v in self.items()})
    
    def size(self, idx, key=None):
        assert idx == 0, 'can only get batch size for SensorPack'
        if key is None:
            key = list(self.keys())[0]
        return self[key].size(idx)

    def dim2_att(self, val1, val2, padd=0):
        return SensorPack({k: v[val1:val1+1,val2[0]:val2[1]] for k, v in self.items()})

    def collect_tower(self, val1s, val2s, max_length):
        sensor_dict = {}
        for k, v in self.items():
            temp_values = []
            for l in range(len(val1s)):
                va = v[val1s[l]:val1s[l]+1, val2s[l][0]:val2s[l][1]]
                va = torch.cat((va, torch.zeros(1, max_length - va.shape[1], *va.shape[2:])),1)
                temp_values.append(va)
            sensor_dict[k] = torch.cat(temp_values,0)
        return SensorPack(sensor_dict)