@description('Name for the Form Recognizer resource.')
param acrName string
@description('SKU for Form Recognizer')
@allowed(['Basic', 'Standard', 'Premium'])
param acrSKU string = 'Standard'
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: toLower('acr${acrName}')
  location: location
  sku: {
    name: acrSKU
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    adminUserEnabled: false
  }
  tags: {
    language: language
  }
}

output acrNameOutput string = acr.name
output acrLoginServerOutput string = acr.properties.loginServer
