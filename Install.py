import pip

# Указываем пакеты для установки
packages = ['scikit-learn']

# Устанавливаем пакеты
for package in packages:
    pip.main(['install', package])
