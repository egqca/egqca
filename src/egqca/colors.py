
# coding: utf-8


import matplotlib.pyplot as plt
import matplotlib


class Palette:

	def __init__(self, num = 10 , alpha = 0.75 , palette_name='rainbow'):
		self.num = num
		self.alpha = alpha
		self.palette_name = palette_name
		self.colors = []


		try:
			cmap = getattr(matplotlib.cm, self.palette_name) # equivalent to 'cmap = matplotlib.cm.palette'

		except:
			print('Invalid color palette\n')

		else:
			norm = matplotlib.colors.Normalize(vmin=0, vmax=self.num)

			aux = [cmap(norm(value), alpha = self.alpha) for value in range(0,self.num)]

            # round RGB to 3 decimal places
			for item in aux:
				self.colors.append([ round(v,3) for v in item ])


	def get_colors(self):
		if self.colors != None:
			# print('\n --------------------------------------------------Codigos RGBA------------------------------------------------- \n')
			# for c in self.colors:
			# 	print(c)

			return self.colors
		else:
			print("color palette is empty!\n")


	def graph(self):
		if self.colors != None:
			fig, ax = plt.subplots(figsize=(14,8))
			plt.imshow([self.colors])

			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['bottom'].set_visible(False)

			plt.tight_layout()
			plt.title("Palette: {}".format(self.palette_name) , fontsize=12, weight='bold')
			plt.yticks([])
			plt.xticks(range(0,self.num), self.colors , rotation='vertical' , fontsize=12, weight='bold')
			plt.show()

		else:
			print("color palette is empty!\n")


# -------------------------------------------------------------------------------------------------------


#print("\nSEE ALL PALETTES ===> https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html \n")


def program():
	#parametros ------------------------------------------
	#(ex palettes: hsv,winter,rainbow ,plasma, inferno, flag, cool, PiYG, etc...)

	try:
		palette_name=str(input('Name of color palette(case sensitive): \n'))
	except:
		palette_name = None

	try:
		num=int(input('Number of colors \n'))
	except:
		num=None

	try:
		alpha=float(input('alpha (from 0 to 1): \n'))
	except:
		alpha=None

	#------------------------------------------------------
	if palette_name != None and num != None and alpha != None:
		# instanciar objeto
		palette = Palette(num, alpha, palette_name)

		# setar as cores
		palette.set_colors()

		# plotar grafico
		palette.graph()

		# obter os codigos das cores
		palette.get_colors()

		# deletar objeto
		del palette

	else:
		print('Using the default palette \n')
		# instanciar objeto
		palette = Palette()

		# setar as cores
		palette.set_colors()

		# plotar grafico
		palette.graph()

		# obter os codigos das cores
		palette.get_colors()

		# deletar objeto
		del pallete

#program()
