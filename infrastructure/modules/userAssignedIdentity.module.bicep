@description('Name for the Azure Key Vault')
param userIdentityName string
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string

resource uid 'Microsoft.ManagedIdentity/userAssignedIdentities@2018-11-30' = {
  name: toLower('ui-${userIdentityName}')
  location: location
  tags: {
    language: language
  }
}

output userAssignedIdentityOutput string = uid.id
output userIdentityPrincipalOutput string = uid.properties.principalId
output userIdentityNameOutput string = uid.name
